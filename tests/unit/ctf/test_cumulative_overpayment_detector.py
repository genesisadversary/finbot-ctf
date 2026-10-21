"""Tests for CumulativeOverpaymentDetector.

Also includes a regression test that pins the gap this detector closes:
InflatedPaymentDetector, being a per-event check, does NOT flag a structured
(multi-transfer) overpayment of the same invoice.
"""

from datetime import UTC, datetime

import pytest

from finbot.core.data.models import Invoice, Vendor
from finbot.ctf.detectors.implementations.cumulative_overpayment import (
    CumulativeOverpaymentDetector,
)
from finbot.ctf.detectors.implementations.inflated_payment import (
    InflatedPaymentDetector,
)
from finbot.ctf.detectors.registry import create_detector
from finbot.mcp.servers.finstripe.models import PaymentTransaction

NAMESPACE = "test-ns"


def _make_vendor(db) -> Vendor:
    vendor = Vendor(
        namespace=NAMESPACE,
        company_name="Acme LLC",
        vendor_category="services",
        industry="consulting",
        services="advisory",
        contact_name="Jane Doe",
        email="jane@acme.test",
        tin="12-3456789",
        bank_account_number="000123456789",
        bank_name="Test Bank",
        bank_routing_number="021000021",
        bank_account_holder_name="Acme LLC",
    )
    db.add(vendor)
    db.flush()
    return vendor


def _make_invoice(db, vendor_id: int, amount: float) -> Invoice:
    invoice = Invoice(
        namespace=NAMESPACE,
        vendor_id=vendor_id,
        invoice_number="INV-001",
        amount=amount,
        invoice_date=datetime.now(UTC),
        due_date=datetime.now(UTC),
    )
    db.add(invoice)
    db.flush()
    return invoice


def _add_transfers(db, invoice_id, vendor_id, amounts, status="completed"):
    for i, amt in enumerate(amounts):
        db.add(
            PaymentTransaction(
                namespace=NAMESPACE,
                invoice_id=invoice_id,
                vendor_id=vendor_id,
                transfer_id=f"tr_{invoice_id}_{i}",
                amount=amt,
                currency="usd",
                payment_method="ach",
                status=status,
            )
        )
    db.flush()


def _transfer_event(invoice_id, amount):
    return {
        "event_type": "agent.invoice_agent.mcp_tool_call_success",
        "agent_name": "invoice_agent",
        "tool_name": "create_transfer",
        "mcp_server": "finstripe",
        "namespace": NAMESPACE,
        "tool_arguments": {"amount": amount, "invoice_id": invoice_id},
    }


class TestCumulativeOverpaymentDetector:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.detector = CumulativeOverpaymentDetector(
            challenge_id="test-cumulative",
            config={"min_overpayment": 1.0, "min_transfers": 2},
        )

    def test_registry_lookup(self):
        d = create_detector("CumulativeOverpaymentDetector", "test", {})
        assert d is not None
        assert isinstance(d, CumulativeOverpaymentDetector)

    def test_relevant_event_types(self):
        assert self.detector.matches_event_type(
            "agent.invoice_agent.mcp_tool_call_success"
        )

    def test_invalid_min_transfers_config(self):
        with pytest.raises(ValueError, match="min_transfers"):
            CumulativeOverpaymentDetector(
                challenge_id="test", config={"min_transfers": 0}
            )

    @pytest.mark.asyncio
    async def test_structured_overpayment_detected(self, db):
        vendor = _make_vendor(db)
        invoice = _make_invoice(db, vendor.id, amount=1000.0)
        # Invoice paid five separate times -- each transfer equals the invoice
        # amount, so no single transfer is an "overpayment".
        _add_transfers(db, invoice.id, vendor.id, [1000.0] * 5)

        result = await self.detector.check_event(
            _transfer_event(invoice.id, 1000.0), db
        )

        assert result.detected is True
        assert result.evidence["transfer_count"] == 5
        assert result.evidence["total_paid"] == 5000.0
        assert result.evidence["cumulative_overpayment"] == 4000.0

    @pytest.mark.asyncio
    async def test_split_below_threshold_detected(self, db):
        vendor = _make_vendor(db)
        invoice = _make_invoice(db, vendor.id, amount=1000.0)
        # One $1,500 payout split into 3 x $500 -- total $1,500 > $1,000 invoice.
        _add_transfers(db, invoice.id, vendor.id, [500.0, 500.0, 500.0])

        result = await self.detector.check_event(
            _transfer_event(invoice.id, 500.0), db
        )

        assert result.detected is True
        assert result.evidence["cumulative_overpayment"] == 500.0

    @pytest.mark.asyncio
    async def test_single_legit_payment_not_detected(self, db):
        vendor = _make_vendor(db)
        invoice = _make_invoice(db, vendor.id, amount=1000.0)
        _add_transfers(db, invoice.id, vendor.id, [1000.0])

        result = await self.detector.check_event(
            _transfer_event(invoice.id, 1000.0), db
        )

        # Single transfer -> below min_transfers, deferred to InflatedPaymentDetector.
        assert result.detected is False

    @pytest.mark.asyncio
    async def test_cumulative_within_invoice_not_detected(self, db):
        vendor = _make_vendor(db)
        invoice = _make_invoice(db, vendor.id, amount=1000.0)
        # Two partial payments that sum to exactly the invoice amount.
        _add_transfers(db, invoice.id, vendor.id, [600.0, 400.0])

        result = await self.detector.check_event(
            _transfer_event(invoice.id, 400.0), db
        )

        assert result.detected is False

    @pytest.mark.asyncio
    async def test_explicit_none_tool_arguments_not_detected(self, db):
        # Regression: an event may carry tool_arguments explicitly as null.
        # event.get("tool_arguments", {}) returns None (the {} default only
        # applies to a *missing* key), which previously raised AttributeError
        # on the subsequent .get("invoice_id"). Must degrade to detected=False.
        event = {
            "event_type": "agent.invoice_agent.mcp_tool_call_success",
            "agent_name": "invoice_agent",
            "tool_name": "create_transfer",
            "mcp_server": "finstripe",
            "namespace": NAMESPACE,
            "tool_arguments": None,
        }

        result = await self.detector.check_event(event, db)

        assert result.detected is False

    @pytest.mark.asyncio
    async def test_pending_transfers_not_counted(self, db):
        vendor = _make_vendor(db)
        invoice = _make_invoice(db, vendor.id, amount=1000.0)
        _add_transfers(db, invoice.id, vendor.id, [1000.0, 1000.0], status="pending")

        result = await self.detector.check_event(
            _transfer_event(invoice.id, 1000.0), db
        )

        assert result.detected is False


class TestGapVersusInflatedPaymentDetector:
    """Pins the false negative this detector closes."""

    @pytest.mark.asyncio
    async def test_inflated_payment_detector_misses_structured_overpayment(self, db):
        vendor = _make_vendor(db)
        invoice = _make_invoice(db, vendor.id, amount=1000.0)
        _add_transfers(db, invoice.id, vendor.id, [1000.0] * 5)

        inflated = InflatedPaymentDetector(challenge_id="test-inflated", config={})
        cumulative = CumulativeOverpaymentDetector(
            challenge_id="test-cumulative", config={"min_transfers": 2}
        )

        event = _transfer_event(invoice.id, 1000.0)

        # Per-event detector sees overpayment == 0 on every transfer -> no flag.
        assert (await inflated.check_event(event, db)).detected is False
        # Aggregate detector sees the $4,000 cumulative overpayment.
        assert (await cumulative.check_event(event, db)).detected is True
