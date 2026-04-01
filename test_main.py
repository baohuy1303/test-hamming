"""
Pytest tests for parse_menu and validate_order.

Requires a valid OPENAI_API_KEY environment variable.
Run with: pytest test_main.py -v
"""

import pytest
from main import (
    DEFAULT_TAX_RATE,
    Coupon,
    Menu,
    MenuItem,
    Modifier,
    ModifierGroup,
    ModifierOption,
    OrderDecision,
    OrderItem,
    _apply_coupons,
    _validate_menu_prices,
    _validate_modifiers,
    parse_menu,
    validate_order,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_MENU_TEXT = """
Cheeseburger - Small $6.99, Medium $8.99, Large $10.99
  Bun Type (pick 1, required): regular bun, gluten-free bun (+$1.00), pretzel bun (+$0.75)
    Note: gluten-free bun and pretzel bun cannot be selected together
  Toppings (optional, max 3): no pickles, extra cheese (+$0.50), bacon (+$1.50)

Classic Fries - Small $2.49, Medium $3.49, Large $4.49

Soft Drink - Small $1.49, Medium $1.99, Large $2.49
  Drink Choice (pick 1, required): Coke, Diet Coke, Sprite, Lemonade

Combo Meal (Cheeseburger + Fries + Drink) - Medium $12.99, Large $14.99
  Note: coupon codes not valid on combo items

Coupons:
  SPRING10 - 10% off your order (not valid on combo items, cannot combine with other coupons)
  FREEFRIES - Free small fries with any burger purchase (stackable)
  BOGO - Buy one get one 50% off on burgers (cannot combine with SPRING10)
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
            assert item.sku
            assert item.unitPrice > 0

    def test_cheeseburger_sizes_present(self, parsed_menu: Menu):
        cb_items = [
            item for item in parsed_menu.normalizedMenu
            if "cheeseburger" in item.name.lower()
        ]
        sizes = {item.size.lower() for item in cb_items if item.size}
        assert "small" in sizes and "medium" in sizes and "large" in sizes

    def test_modifier_groups_present(self, parsed_menu: Menu):
        """Cheeseburger should have modifierGroups (not flat modifiers)."""
        cb = next(
            (i for i in parsed_menu.normalizedMenu if "cheeseburger" in i.name.lower()),
            None,
        )
        assert cb is not None
        assert len(cb.modifierGroups) > 0

    def test_bun_type_group_is_required(self, parsed_menu: Menu):
        cb = next(
            (i for i in parsed_menu.normalizedMenu if "cheeseburger" in i.name.lower()),
            None,
        )
        bun_group = next(
            (g for g in cb.modifierGroups if "bun" in g.group.lower()), None
        )
        assert bun_group is not None
        assert bun_group.required is True
        assert bun_group.maxSelections == 1

    def test_gluten_free_has_surcharge(self, parsed_menu: Menu):
        cb = next(
            (i for i in parsed_menu.normalizedMenu if "cheeseburger" in i.name.lower()),
            None,
        )
        all_opts = [opt for g in cb.modifierGroups for opt in g.options]
        gf = next((o for o in all_opts if "gluten" in o.name.lower()), None)
        assert gf is not None
        assert gf.surcharge >= 1.0

    def test_gluten_free_conflicts_with_pretzel(self, parsed_menu: Menu):
        cb = next(
            (i for i in parsed_menu.normalizedMenu if "cheeseburger" in i.name.lower()),
            None,
        )
        bun_group = next(
            (g for g in cb.modifierGroups if "bun" in g.group.lower()), None
        )
        gf = next((o for o in bun_group.options if "gluten" in o.name.lower()), None)
        pretzel = next((o for o in bun_group.options if "pretzel" in o.name.lower()), None)
        assert gf is not None and pretzel is not None
        # At least one direction of the conflict should be declared
        has_conflict = (
            pretzel.name.lower() in [c.lower() for c in gf.conflicts]
            or gf.name.lower() in [c.lower() for c in pretzel.conflicts]
        )
        assert has_conflict, "gluten-free and pretzel bun should conflict"

    def test_drink_choice_required_on_soft_drink(self, parsed_menu: Menu):
        sd = next(
            (i for i in parsed_menu.normalizedMenu if "soft drink" in i.name.lower()),
            None,
        )
        assert sd is not None
        drink_group = next(
            (g for g in sd.modifierGroups if "drink" in g.group.lower() or "choice" in g.group.lower()),
            None,
        )
        assert drink_group is not None
        assert drink_group.required is True

    def test_combo_includes_populated(self, parsed_menu: Menu):
        combo = next(
            (i for i in parsed_menu.normalizedMenu if "combo" in i.name.lower()),
            None,
        )
        assert combo is not None
        assert len(combo.comboIncludes) > 0

    def test_coupons_parsed(self, parsed_menu: Menu):
        assert len(parsed_menu.coupons) >= 3
        codes = {c.code.upper() for c in parsed_menu.coupons}
        assert "SPRING10" in codes
        assert "FREEFRIES" in codes
        assert "BOGO" in codes

    def test_spring10_is_not_stackable(self, parsed_menu: Menu):
        spring = next(c for c in parsed_menu.coupons if c.code.upper() == "SPRING10")
        assert spring.stackable is False

    def test_bogo_excludes_spring10(self, parsed_menu: Menu):
        bogo = next(c for c in parsed_menu.coupons if c.code.upper() == "BOGO")
        assert "SPRING10" in [e.upper() for e in bogo.excludes]

    def test_spring10_excludes_items_combo(self, parsed_menu: Menu):
        spring = next(c for c in parsed_menu.coupons if c.code.upper() == "SPRING10")
        # Should exclude combo items via excludesItems
        all_text = " ".join(spring.excludesItems + spring.restrictions).lower()
        assert "combo" in all_text

    def test_prices_are_rounded(self, parsed_menu: Menu):
        for item in parsed_menu.normalizedMenu:
            assert item.unitPrice == round(item.unitPrice, 2)

    def test_no_price_warnings_on_clean_menu(self, parsed_menu: Menu):
        assert len(parsed_menu.priceWarnings) == 0


# ---------------------------------------------------------------------------
# Tests — price validator (deterministic, no LLM)
# ---------------------------------------------------------------------------

class TestPriceValidator:
    def test_detects_conflicting_prices(self):
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
        assert len(_validate_menu_prices(menu)) == 0

    def test_flags_zero_price(self):
        menu = Menu(normalizedMenu=[
            MenuItem(sku="X-1", name="Free Thing", size=None, unitPrice=0.0),
        ])
        warnings = _validate_menu_prices(menu)
        assert len(warnings) == 1
        assert "invalid price" in warnings[0].message.lower()


# ---------------------------------------------------------------------------
# Tests — modifier DSL validation (deterministic, no LLM)
# ---------------------------------------------------------------------------

class TestModifierDSL:
    """Unit tests for _validate_modifiers — pure Python, no API calls."""

    @pytest.fixture()
    def burger_item(self) -> MenuItem:
        return MenuItem(
            sku="CB-MED",
            name="Cheeseburger",
            size="Medium",
            unitPrice=8.99,
            modifierGroups=[
                ModifierGroup(
                    group="Bun Type",
                    required=True,
                    minSelections=1,
                    maxSelections=1,
                    options=[
                        ModifierOption(name="regular bun", surcharge=0.0),
                        ModifierOption(
                            name="gluten-free bun",
                            surcharge=1.00,
                            conflicts=["pretzel bun"],
                        ),
                        ModifierOption(
                            name="pretzel bun",
                            surcharge=0.75,
                            conflicts=["gluten-free bun"],
                        ),
                    ],
                ),
                ModifierGroup(
                    group="Toppings",
                    required=False,
                    minSelections=0,
                    maxSelections=3,
                    options=[
                        ModifierOption(name="no pickles", surcharge=0.0),
                        ModifierOption(name="extra cheese", surcharge=0.50),
                        ModifierOption(name="bacon", surcharge=1.50),
                    ],
                ),
            ],
        )

    def test_valid_selection(self, burger_item: MenuItem):
        mods = [Modifier(name="gluten-free bun"), Modifier(name="no pickles")]
        validated, issues, repairs = _validate_modifiers(mods, burger_item)
        assert len(validated) == 2
        assert len(issues) == 0

    def test_conflict_detected(self, burger_item: MenuItem):
        mods = [Modifier(name="gluten-free bun"), Modifier(name="pretzel bun")]
        validated, issues, repairs = _validate_modifiers(mods, burger_item)
        assert len(validated) == 1  # only first one accepted
        assert any("conflicts" in i.lower() for i in issues)
        assert len(repairs) > 0

    def test_max_cardinality_enforced(self, burger_item: MenuItem):
        """Bun Type has maxSelections=1, so only 1 bun should be accepted."""
        mods = [Modifier(name="regular bun"), Modifier(name="gluten-free bun")]
        validated, issues, repairs = _validate_modifiers(mods, burger_item)
        buns = [m for m in validated if "bun" in m.name.lower()]
        assert len(buns) == 1
        assert any("max" in i.lower() for i in issues)

    def test_required_group_missing(self, burger_item: MenuItem):
        """No bun type selected — should flag required group."""
        mods = [Modifier(name="no pickles")]
        validated, issues, repairs = _validate_modifiers(mods, burger_item)
        assert any("requires" in i.lower() and "bun" in i.lower() for i in issues)
        assert len(repairs) > 0

    def test_unknown_modifier_rejected(self, burger_item: MenuItem):
        mods = [Modifier(name="truffle oil")]
        validated, issues, repairs = _validate_modifiers(mods, burger_item)
        assert len(validated) == 0
        assert any("not available" in i.lower() for i in issues)

    def test_surcharge_from_menu(self, burger_item: MenuItem):
        """Surcharge should come from menu, not from the customer's input."""
        mods = [Modifier(name="extra cheese", surcharge=999.99)]
        validated, issues, repairs = _validate_modifiers(mods, burger_item)
        assert len(validated) == 1
        assert validated[0].surcharge == 0.50  # menu truth, not 999.99


# ---------------------------------------------------------------------------
# Tests — coupon stacking engine (deterministic, no LLM)
# ---------------------------------------------------------------------------

class TestCouponStacking:
    """Unit tests for _apply_coupons — pure Python, no API calls."""

    @pytest.fixture()
    def coupons(self) -> dict[str, Coupon]:
        return {
            "SPRING10": Coupon(
                code="SPRING10",
                discountType="percent",
                discountValue=10.0,
                stackable=False,
                excludesItems=["Combo"],
            ),
            "FREEFRIES": Coupon(
                code="FREEFRIES",
                discountType="flat",
                discountValue=2.49,
                stackable=True,
                appliesTo=["Fries"],
            ),
            "BOGO": Coupon(
                code="BOGO",
                discountType="percent",
                discountValue=50.0,
                stackable=True,
                excludes=["SPRING10"],
                appliesTo=["Cheeseburger"],
            ),
        }

    @pytest.fixture()
    def items(self) -> list[OrderItem]:
        return [
            OrderItem(sku="CB-MED", name="Cheeseburger", size="Medium", unitPrice=8.99),
        ]

    def test_single_valid_coupon(self, coupons, items):
        applied, disc, issues = _apply_coupons(["SPRING10"], coupons, items, 8.99)
        assert len(applied) == 1
        assert applied[0].code == "SPRING10"
        assert disc == round(8.99 * 0.10, 2)
        assert len(issues) == 0

    def test_invalid_coupon_code(self, coupons, items):
        applied, disc, issues = _apply_coupons(["FAKE123"], coupons, items, 8.99)
        assert len(applied) == 0
        assert disc == 0.0
        assert any("not a valid" in i.lower() for i in issues)

    def test_non_stackable_blocks_second(self, coupons, items):
        """SPRING10 is non-stackable, so FREEFRIES cannot be added after it."""
        applied, disc, issues = _apply_coupons(
            ["SPRING10", "FREEFRIES"], coupons, items, 8.99
        )
        assert len(applied) == 1
        assert applied[0].code == "SPRING10"
        assert any("non-stackable" in i.lower() or "cannot" in i.lower() for i in issues)

    def test_mutual_exclusion(self, coupons, items):
        """BOGO excludes SPRING10."""
        applied, disc, issues = _apply_coupons(
            ["BOGO", "SPRING10"], coupons, items, 8.99
        )
        # BOGO applies first; SPRING10 is non-stackable and can't add after BOGO
        # or SPRING10 excludes BOGO — either way only 1 should apply
        assert len(applied) <= 1
        assert len(issues) > 0

    def test_excludes_items_blocks_combo(self, coupons):
        combo_items = [
            OrderItem(sku="CM-MED", name="Combo Meal", size="Medium", unitPrice=12.99),
        ]
        applied, disc, issues = _apply_coupons(
            ["SPRING10"], coupons, combo_items, 12.99
        )
        assert len(applied) == 0
        assert any("not valid on" in i.lower() for i in issues)

    def test_applies_to_filters_eligible_total(self, coupons):
        """FREEFRIES only applies to fries items."""
        items_with_fries = [
            OrderItem(sku="CB-MED", name="Cheeseburger", size="Medium", unitPrice=8.99),
            OrderItem(sku="FF-SM", name="Classic Fries", size="Small", unitPrice=2.49),
        ]
        applied, disc, issues = _apply_coupons(
            ["FREEFRIES"], coupons, items_with_fries, 11.48
        )
        assert len(applied) == 1
        assert applied[0].discountAmount == 2.49  # flat capped at fries price

    def test_stackable_coupons_combine(self, coupons):
        """BOGO + FREEFRIES should both apply (both stackable, no mutual exclusion)."""
        items = [
            OrderItem(sku="CB-MED", name="Cheeseburger", size="Medium", unitPrice=8.99),
            OrderItem(sku="FF-SM", name="Classic Fries", size="Small", unitPrice=2.49),
        ]
        applied, disc, issues = _apply_coupons(
            ["BOGO", "FREEFRIES"], coupons, items, 11.48
        )
        assert len(applied) == 2
        codes = {a.code for a in applied}
        assert "BOGO" in codes and "FREEFRIES" in codes


# ---------------------------------------------------------------------------
# Tests — validate_order (full 4-stage pipeline, requires API)
# ---------------------------------------------------------------------------

class TestValidateOrder:
    ORDER_TEXT = (
        "I'd like a medium cheeseburger on a gluten-free bun with no pickles, "
        "and can I use coupon SPRING10?"
    )

    @pytest.fixture(scope="class")
    def decision(self, parsed_menu: Menu) -> OrderDecision:
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
        sku_prices = {
            item.sku.upper(): item.unitPrice for item in parsed_menu.normalizedMenu
        }
        for item in decision.normalizedOrder:
            assert item.unitPrice == sku_prices.get(item.sku.upper(), item.unitPrice)

    def test_subtotal_is_correct(self, decision: OrderDecision):
        expected = 0.0
        for item in decision.normalizedOrder:
            mod_total = sum(m.surcharge for m in item.modifiers)
            expected += (item.unitPrice + mod_total) * item.qty
        expected = round(expected, 2)
        assert decision.total.subtotal == expected

    def test_coupon_applied(self, decision: OrderDecision):
        assert len(decision.total.couponsApplied) > 0
        codes = {c.code.upper() for c in decision.total.couponsApplied}
        assert "SPRING10" in codes
        assert decision.total.discount > 0

    def test_discount_amount_is_exact(self, decision: OrderDecision):
        for ca in decision.total.couponsApplied:
            if ca.discountType == "percent":
                expected = round(
                    decision.total.subtotal * ca.discountValue / 100, 2
                )
                assert ca.discountAmount == expected

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
            assert val == round(val, 2), f"{field} not rounded: {val}"

    def test_confirmation_text_not_empty(self, decision: OrderDecision):
        assert decision.confirmationText.strip()

    def test_confirmation_mentions_total(self, decision: OrderDecision):
        gt = f"{decision.total.grandTotal:.2f}"
        assert gt in decision.confirmationText

    def test_accepts_dict_order(self, parsed_menu: Menu):
        order_dict = {
            "items": [{"name": "Medium Cheeseburger", "modifiers": ["no pickles"]}]
        }
        result = validate_order(parsed_menu, order_dict)
        assert isinstance(result, OrderDecision)


# ---------------------------------------------------------------------------
# Tests — multi-language (requires API)
# ---------------------------------------------------------------------------

class TestMultiLanguage:
    SPANISH_MENU = """
    Hamburguesa con queso - Pequena $6.99, Mediana $8.99, Grande $10.99
      Opciones: sin pepinillos, queso extra (+$0.50)

    Papas fritas - Pequena $2.49, Mediana $3.49, Grande $4.49
    """

    def test_spanish_menu_parses_to_english(self):
        menu = parse_menu(self.SPANISH_MENU, translate=True)
        assert len(menu.normalizedMenu) > 0
        # Items should be in English
        all_names = " ".join(i.name.lower() for i in menu.normalizedMenu)
        assert "burger" in all_names or "cheese" in all_names or "fries" in all_names
