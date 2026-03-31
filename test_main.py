"""
Pytest test for parse_menu and validate_order.

Requires a valid OPENAI_API_KEY environment variable.
Run with: pytest test_main.py -v
"""

import pytest
from main import (
    DEFAULT_TAX_RATE,
    Menu,
    MenuItem,
    Modifier,
    OrderDecision,
    _validate_menu_prices,
    parse_menu,
    validate_order,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_MENU_TEXT = """
Cheeseburger - Small $6.99, Medium $8.99, Large $10.99
  Options: no pickles, extra cheese (+$0.50), gluten-free bun (+$1.00)

Classic Fries - Small $2.49, Medium $3.49, Large $4.49

Soft Drink - Small $1.49, Medium $1.99, Large $2.49
  Options: Coke, Diet Coke, Sprite, Lemonade

Combo Meal (Cheeseburger + Fries + Drink) - Medium $12.99, Large $14.99
  Note: coupon codes not valid on combo items

Coupons:
  SPRING10 - 10% off your order (not valid on combo items)
  FREEFRIES - Free small fries with any burger purchase
"""


@pytest.fixture(scope="module")
def parsed_menu() -> Menu:
    """Parse the sample menu once for the whole test module."""
    return parse_menu(SAMPLE_MENU_TEXT)


# ---------------------------------------------------------------------------
# Tests — parse_menu
# ---------------------------------------------------------------------------

class TestParseMenu:
    def test_returns_menu_instance(self, parsed_menu: Menu):
        assert isinstance(parsed_menu, Menu)

    def test_has_items(self, parsed_menu: Menu):
        assert len(parsed_menu.normalizedMenu) > 0

    def test_all_items_have_sku_and_price(self, parsed_menu: Menu):
        for item in parsed_menu.normalizedMenu:
            assert isinstance(item, MenuItem)
            assert item.sku, "SKU must not be empty"
            assert item.unitPrice > 0, f"{item.name} must have a positive price"

    def test_cheeseburger_sizes_present(self, parsed_menu: Menu):
        cb_items = [
            item for item in parsed_menu.normalizedMenu
            if "cheeseburger" in item.name.lower()
        ]
        sizes = {item.size.lower() for item in cb_items if item.size}
        assert "small" in sizes and "medium" in sizes and "large" in sizes

    def test_modifier_has_surcharge(self, parsed_menu: Menu):
        all_mods = [
            mod for item in parsed_menu.normalizedMenu for mod in item.modifiers
        ]
        assert any(mod.surcharge > 0 for mod in all_mods), (
            "Expected at least one modifier with surcharge > 0"
        )

    def test_gluten_free_modifier_captured(self, parsed_menu: Menu):
        cb_items = [
            item for item in parsed_menu.normalizedMenu
            if "cheeseburger" in item.name.lower()
        ]
        assert cb_items, "No cheeseburger items found"
        all_mod_names = [mod.name.lower() for item in cb_items for mod in item.modifiers]
        assert any("gluten" in m for m in all_mod_names)

    def test_combo_includes_populated(self, parsed_menu: Menu):
        combo_items = [
            item for item in parsed_menu.normalizedMenu
            if "combo" in item.name.lower()
        ]
        assert combo_items, "No combo items found in menu"
        for combo in combo_items:
            assert len(combo.comboIncludes) > 0

    def test_combo_has_notes(self, parsed_menu: Menu):
        combo_items = [
            item for item in parsed_menu.normalizedMenu
            if "combo" in item.name.lower()
        ]
        all_notes = [n.lower() for combo in combo_items for n in combo.notes]
        assert any("coupon" in n for n in all_notes)

    def test_coupons_parsed(self, parsed_menu: Menu):
        assert len(parsed_menu.coupons) >= 2
        codes = {c.code.upper() for c in parsed_menu.coupons}
        assert "SPRING10" in codes
        assert "FREEFRIES" in codes

    def test_coupon_has_restriction(self, parsed_menu: Menu):
        spring = next(
            (c for c in parsed_menu.coupons if c.code.upper() == "SPRING10"), None
        )
        assert spring is not None
        assert spring.discountType == "percent"
        assert spring.discountValue == 10.0
        assert any("combo" in r.lower() for r in spring.restrictions)

    def test_prices_are_rounded(self, parsed_menu: Menu):
        for item in parsed_menu.normalizedMenu:
            assert item.unitPrice == round(item.unitPrice, 2)
            for mod in item.modifiers:
                assert mod.surcharge == round(mod.surcharge, 2)

    def test_no_price_warnings_on_clean_menu(self, parsed_menu: Menu):
        assert len(parsed_menu.priceWarnings) == 0


# ---------------------------------------------------------------------------
# Tests — price validator
# ---------------------------------------------------------------------------

class TestPriceValidator:
    def test_detects_conflicting_prices(self):
        """Manually create a menu with duplicate items at different prices."""
        menu = Menu(normalizedMenu=[
            MenuItem(sku="SD-LRG-1", name="Soft Drink", size="Large", unitPrice=2.49),
            MenuItem(sku="SD-LRG-2", name="Soft Drink", size="Large", unitPrice=2.59),
        ])
        warnings = _validate_menu_prices(menu)
        assert len(warnings) == 1
        assert "2.49" in warnings[0].message
        assert "2.59" in warnings[0].message

    def test_no_warning_for_consistent_prices(self):
        menu = Menu(normalizedMenu=[
            MenuItem(sku="CB-SM", name="Burger", size="Small", unitPrice=5.99),
            MenuItem(sku="CB-MD", name="Burger", size="Medium", unitPrice=7.99),
        ])
        warnings = _validate_menu_prices(menu)
        assert len(warnings) == 0

    def test_flags_zero_price(self):
        menu = Menu(normalizedMenu=[
            MenuItem(sku="X-1", name="Free Thing", size=None, unitPrice=0.0),
        ])
        warnings = _validate_menu_prices(menu)
        assert len(warnings) == 1
        assert "invalid price" in warnings[0].message.lower()


# ---------------------------------------------------------------------------
# Tests — validate_order (4-stage pipeline)
# ---------------------------------------------------------------------------

class TestValidateOrder:
    ORDER_TEXT = (
        "I'd like a medium cheeseburger on a gluten-free bun with no pickles, "
        "and can I use coupon SPRING10?"
    )

    @pytest.fixture(scope="class")
    def decision(self, parsed_menu: Menu) -> OrderDecision:
        """Run validate_order once for all tests in this class."""
        return validate_order(parsed_menu, self.ORDER_TEXT)

    def test_returns_order_decision(self, decision: OrderDecision):
        assert isinstance(decision, OrderDecision)

    def test_normalized_order_not_empty(self, decision: OrderDecision):
        assert len(decision.normalizedOrder) > 0

    def test_order_modifiers_are_objects(self, decision: OrderDecision):
        for item in decision.normalizedOrder:
            for mod in item.modifiers:
                assert isinstance(mod, Modifier)

    def test_prices_match_menu(self, parsed_menu: Menu, decision: OrderDecision):
        """Stage 2 should override prices with menu truth."""
        sku_prices = {item.sku.upper(): item.unitPrice for item in parsed_menu.normalizedMenu}
        for item in decision.normalizedOrder:
            assert item.unitPrice == sku_prices.get(item.sku.upper(), item.unitPrice)

    def test_subtotal_is_correct(self, decision: OrderDecision):
        """Subtotal must equal sum of (unitPrice + surcharges) * qty."""
        expected = 0.0
        for item in decision.normalizedOrder:
            mod_total = sum(m.surcharge for m in item.modifiers)
            expected += (item.unitPrice + mod_total) * item.qty
        expected = round(expected, 2)
        assert decision.total.subtotal == expected

    def test_coupon_applied(self, decision: OrderDecision):
        """SPRING10 is valid on a non-combo cheeseburger — should be applied."""
        assert decision.total.couponApplied is not None
        assert decision.total.couponApplied.code.upper() == "SPRING10"
        assert decision.total.discount > 0

    def test_discount_amount_is_exact(self, decision: OrderDecision):
        """10% of subtotal, rounded to 2dp."""
        if decision.total.couponApplied and decision.total.couponApplied.discountType == "percent":
            expected = round(
                decision.total.subtotal * decision.total.couponApplied.discountValue / 100, 2
            )
            assert decision.total.couponApplied.discountAmount == expected
            assert decision.total.discount == expected

    def test_taxable_amount_is_exact(self, decision: OrderDecision):
        expected = round(decision.total.subtotal - decision.total.discount, 2)
        assert decision.total.taxableAmount == expected

    def test_tax_is_exact(self, decision: OrderDecision):
        expected = round(decision.total.taxableAmount * DEFAULT_TAX_RATE, 2)
        assert decision.total.tax == expected

    def test_grand_total_is_exact(self, decision: OrderDecision):
        expected = round(decision.total.taxableAmount + decision.total.tax, 2)
        assert decision.total.grandTotal == expected

    def test_all_totals_rounded_to_2dp(self, decision: OrderDecision):
        for field in ["subtotal", "discount", "taxableAmount", "tax", "grandTotal"]:
            val = getattr(decision.total, field)
            assert val == round(val, 2), f"{field} not rounded to 2dp: {val}"

    def test_confirmation_text_not_empty(self, decision: OrderDecision):
        assert decision.confirmationText.strip()

    def test_confirmation_mentions_total(self, decision: OrderDecision):
        gt = f"{decision.total.grandTotal:.2f}"
        assert gt in decision.confirmationText, (
            f"Confirmation should mention grand total ${gt}"
        )

    def test_accepts_dict_order(self, parsed_menu: Menu):
        order_dict = {
            "items": [{"name": "Medium Cheeseburger", "modifiers": ["no pickles"]}]
        }
        result = validate_order(parsed_menu, order_dict)
        assert isinstance(result, OrderDecision)
