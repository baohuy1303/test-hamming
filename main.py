from dotenv import load_dotenv
import json
import os
from collections import defaultdict
from typing import Optional, Union

from openai import OpenAI
from pydantic import BaseModel, Field, model_validator

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TAX_RATE = 0.0825
CONFIDENCE_THRESHOLD = 0.85
DEFAULT_SEED = 42

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class Modifier(BaseModel):
    name: str
    surcharge: float = 0.0

    @model_validator(mode="after")
    def _round(self):
        self.surcharge = round(self.surcharge, 2)
        return self


class Coupon(BaseModel):
    code: str
    discountType: str  # "percent" | "flat"
    discountValue: float
    restrictions: list[str] = Field(default_factory=list)


class PriceInconsistency(BaseModel):
    itemName: str
    size: Optional[str]
    prices: list[float]
    message: str


class MenuItem(BaseModel):
    sku: str
    name: str
    size: Optional[str] = None
    modifiers: list[Modifier] = Field(default_factory=list)
    comboIncludes: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    qty: int = 1
    unitPrice: float

    @model_validator(mode="after")
    def _round(self):
        self.unitPrice = round(self.unitPrice, 2)
        return self


class Menu(BaseModel):
    normalizedMenu: list[MenuItem]
    coupons: list[Coupon] = Field(default_factory=list)
    priceWarnings: list[PriceInconsistency] = Field(default_factory=list)


class OrderItem(BaseModel):
    sku: str
    name: str
    size: Optional[str] = None
    modifiers: list[Modifier] = Field(default_factory=list)
    qty: int = 1
    unitPrice: float

    @model_validator(mode="after")
    def _round(self):
        self.unitPrice = round(self.unitPrice, 2)
        return self


class CouponApplied(BaseModel):
    code: str
    discountType: str  # "percent" | "flat"
    discountValue: float
    discountAmount: float  # actual dollar amount deducted

    @model_validator(mode="after")
    def _round(self):
        self.discountAmount = round(self.discountAmount, 2)
        return self


class Total(BaseModel):
    subtotal: float
    couponApplied: Optional[CouponApplied] = None
    discount: float = 0.0
    taxableAmount: float
    tax: float
    grandTotal: float

    @model_validator(mode="after")
    def _round(self):
        self.subtotal = round(self.subtotal, 2)
        self.discount = round(self.discount, 2)
        self.taxableAmount = round(self.taxableAmount, 2)
        self.tax = round(self.tax, 2)
        self.grandTotal = round(self.grandTotal, 2)
        return self


class OrderDecision(BaseModel):
    normalizedOrder: list[OrderItem]
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    total: Total
    confirmationText: str


# Stage 1 intermediate model — what the LLM returns (no math)
class RawOrderExtraction(BaseModel):
    items: list[OrderItem]
    couponCodeMentioned: Optional[str] = None
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


# Stage 4 judge model
class JudgeResult(BaseModel):
    confidence: float
    issues: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# parse_menu
# ---------------------------------------------------------------------------

_PARSE_MENU_SYSTEM = """\
You are a helpful restaurant employee assistant. Your job is to parse a \
natural-language menu description into a structured JSON format.

Rules:
- Assign a short, readable SKU to every item (e.g. "CB-MED" for Medium Cheeseburger).
- Capture all size variants as separate entries.
- Modifiers must be objects with "name" and "surcharge" (0.0 if free). Do NOT invent modifiers not mentioned.
- For combo items, list the included item names in "comboIncludes".
- Capture any restrictions or notes in the "notes" array (e.g. "coupon codes not valid on combo items").
- If coupons/promo codes are mentioned, extract them into the top-level "coupons" array.
  Each coupon has: code, discountType ("percent" or "flat"), discountValue (number), restrictions (list of strings).
- If the menu text has spelling errors, normalize the spelling in the output.
- If two prices are listed for the same item/size, include the item once and use the first price mentioned.
- qty is always 1 for a menu entry.
- unitPrice must be a number (no currency symbol).
- Return ONLY valid JSON matching the schema provided.
"""

_PARSE_MENU_SCHEMA = """\
{
  "normalizedMenu": [
    {
      "sku": "string",
      "name": "string",
      "size": "string|null",
      "modifiers": [{"name": "string", "surcharge": 0.00}],
      "comboIncludes": ["string"],
      "notes": ["string"],
      "qty": 1,
      "unitPrice": 0.00
    }
  ],
  "coupons": [
    {
      "code": "string",
      "discountType": "percent|flat",
      "discountValue": 0.00,
      "restrictions": ["string"]
    }
  ]
}"""


def _validate_menu_prices(menu: Menu) -> list[PriceInconsistency]:
    """Detect duplicate (name, size) entries with conflicting prices."""
    warnings: list[PriceInconsistency] = []
    groups: dict[tuple[str, Optional[str]], list[float]] = defaultdict(list)

    for item in menu.normalizedMenu:
        key = (item.name.lower().strip(), (item.size or "").lower().strip() or None)
        groups[key].append(item.unitPrice)

    for (name, size), prices in groups.items():
        unique = sorted(set(prices))
        if len(unique) > 1:
            size_label = size.title() if size else "default"
            price_strs = " and ".join(f"${p:.2f}" for p in unique)
            warnings.append(PriceInconsistency(
                itemName=name.title(),
                size=size,
                prices=unique,
                message=f"{size_label} {name.title()} listed as {price_strs} — pick one",
            ))

    # Flag non-positive prices
    for item in menu.normalizedMenu:
        if item.unitPrice <= 0:
            warnings.append(PriceInconsistency(
                itemName=item.name,
                size=item.size,
                prices=[item.unitPrice],
                message=f"{item.name} ({item.size or 'default'}) has invalid price ${item.unitPrice:.2f}",
            ))

    return warnings


def parse_menu(menu_text: str, seed: int = DEFAULT_SEED) -> Menu:
    """Parse natural-language menu text into a normalized Menu object."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": _PARSE_MENU_SYSTEM},
            {
                "role": "user",
                "content": (
                    "Parse the following menu into the JSON schema.\n\n"
                    f"Schema:\n{_PARSE_MENU_SCHEMA}\n\n"
                    f"Menu:\n{menu_text}"
                ),
            },
        ],
        response_format={"type": "json_object"},
        temperature=0,
        seed=seed,
    )

    raw = response.choices[0].message.content
    data = json.loads(raw)
    menu = Menu.model_validate(data)

    # Run deterministic price validation
    menu.priceWarnings = _validate_menu_prices(menu)

    return menu


# ---------------------------------------------------------------------------
# validate_order — 4-stage pipeline
# ---------------------------------------------------------------------------

# -- Stage 1: LLM Extract --------------------------------------------------

_EXTRACT_SYSTEM = """\
You are a restaurant order parser. Given a menu and a customer's order, extract \
the structured order data. Do NOT calculate totals — just identify items.

Rules:
- Match each requested item to the closest menu item by SKU.
- Include the modifiers the customer asked for (use the modifier names from the menu).
- Set qty based on what the customer asked for.
- Use the unitPrice from the menu for each item (do NOT make up prices).
- If the customer mentions a coupon/promo code, put it in "couponCodeMentioned".
- If an item or modifier doesn't exist on the menu, add it to "issues".
- Suggest relevant upsells or combos in "suggestions".
- Return ONLY valid JSON matching the schema.
"""

_EXTRACT_SCHEMA = """\
{
  "items": [
    {
      "sku": "string",
      "name": "string",
      "size": "string|null",
      "modifiers": [{"name": "string", "surcharge": 0.00}],
      "qty": 1,
      "unitPrice": 0.00
    }
  ],
  "couponCodeMentioned": "string|null",
  "issues": ["string"],
  "suggestions": ["string"]
}"""


def _extract_order(
    menu: Menu,
    order_input: str,
    seed: int,
) -> RawOrderExtraction:
    """Stage 1: Use LLM to extract order items from natural language."""
    menu_json = menu.model_dump_json(indent=2)

    if menu.coupons:
        coupons_section = "Available coupons:\n" + json.dumps(
            [c.model_dump() for c in menu.coupons], indent=2
        )
    else:
        coupons_section = "No coupons are available."

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _EXTRACT_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Menu:\n{menu_json}\n\n"
                    f"{coupons_section}\n\n"
                    f"Customer order:\n{order_input}\n\n"
                    f"Schema:\n{_EXTRACT_SCHEMA}"
                ),
            },
        ],
        response_format={"type": "json_object"},
        temperature=0,
        seed=seed,
    )

    raw = response.choices[0].message.content
    data = json.loads(raw)
    return RawOrderExtraction.model_validate(data)


# -- Stage 2: Python Validate & Calculate ----------------------------------

def _calculate_order(
    menu: Menu,
    extraction: RawOrderExtraction,
    tax_rate: float,
) -> OrderDecision:
    """Stage 2: Pure Python — validate items, apply coupons, compute totals."""
    # Build lookup tables
    sku_lookup: dict[str, MenuItem] = {
        item.sku.upper(): item for item in menu.normalizedMenu
    }
    coupon_lookup: dict[str, Coupon] = {
        c.code.upper(): c for c in menu.coupons
    }

    validated_items: list[OrderItem] = []
    issues = list(extraction.issues)  # start with LLM-detected issues
    suggestions = list(extraction.suggestions)

    subtotal = 0.0

    for item in extraction.items:
        menu_item = sku_lookup.get(item.sku.upper())

        if not menu_item:
            issues.append(f"Item '{item.name}' (SKU: {item.sku}) not found on menu")
            continue

        # Override price with menu truth
        true_price = menu_item.unitPrice

        # Validate and correct modifiers
        menu_mod_lookup = {m.name.lower(): m for m in menu_item.modifiers}
        validated_mods: list[Modifier] = []
        for mod in item.modifiers:
            menu_mod = menu_mod_lookup.get(mod.name.lower())
            if menu_mod:
                validated_mods.append(Modifier(
                    name=menu_mod.name,
                    surcharge=menu_mod.surcharge,
                ))
            else:
                issues.append(
                    f"Modifier '{mod.name}' is not available on {menu_item.name}"
                )

        # Build validated order item
        validated_item = OrderItem(
            sku=menu_item.sku,
            name=menu_item.name,
            size=menu_item.size,
            modifiers=validated_mods,
            qty=item.qty,
            unitPrice=true_price,
        )
        validated_items.append(validated_item)

        # Accumulate subtotal
        mod_surcharges = sum(m.surcharge for m in validated_mods)
        subtotal += (true_price + mod_surcharges) * item.qty

    subtotal = round(subtotal, 2)

    # Coupon validation
    coupon_applied: Optional[CouponApplied] = None
    discount = 0.0

    if extraction.couponCodeMentioned:
        code = extraction.couponCodeMentioned.upper()
        coupon = coupon_lookup.get(code)

        if not coupon:
            issues.append(f"Coupon '{extraction.couponCodeMentioned}' is not a valid coupon code")
        else:
            # Check restrictions
            restriction_hit = False
            for restriction in coupon.restrictions:
                restriction_lower = restriction.lower()
                # Check if any ordered item matches the restriction
                for v_item in validated_items:
                    item_name_lower = v_item.name.lower()
                    # Check combo restriction
                    if "combo" in restriction_lower and "combo" in item_name_lower:
                        issues.append(
                            f"Coupon {coupon.code} is not valid on combo items "
                            f"({v_item.name})"
                        )
                        restriction_hit = True
                        break
                if restriction_hit:
                    break

            if not restriction_hit:
                # Apply the coupon
                if coupon.discountType == "percent":
                    discount = round(subtotal * coupon.discountValue / 100, 2)
                elif coupon.discountType == "flat":
                    discount = round(min(coupon.discountValue, subtotal), 2)

                coupon_applied = CouponApplied(
                    code=coupon.code,
                    discountType=coupon.discountType,
                    discountValue=coupon.discountValue,
                    discountAmount=discount,
                )

    taxable_amount = round(subtotal - discount, 2)
    tax = round(taxable_amount * tax_rate, 2)
    grand_total = round(taxable_amount + tax, 2)

    total = Total(
        subtotal=subtotal,
        couponApplied=coupon_applied,
        discount=discount,
        taxableAmount=taxable_amount,
        tax=tax,
        grandTotal=grand_total,
    )

    return OrderDecision(
        normalizedOrder=validated_items,
        issues=issues,
        suggestions=suggestions,
        total=total,
        confirmationText="",  # placeholder — filled by Stage 3/4
    )


# -- Stage 3: LLM Confirm --------------------------------------------------

_CONFIRM_SYSTEM = """\
You are a friendly, upbeat restaurant counter employee. Based on the order \
details provided, write a short 1-3 sentence confirmation to read back to the \
customer.

Rules:
- List each item with size, modifiers, and quantity.
- If a coupon was applied, mention it by code and the discount amount.
- Always state the grand total with a dollar sign.
- End with a simple yes/no confirmation question.
- Plain text only. No emoji, no markdown, no bullet points.
"""


def _generate_confirmation(
    order: OrderDecision,
    seed: int,
) -> str:
    """Stage 3: Use LLM to generate friendly confirmation text."""
    order_summary = order.model_dump_json(indent=2)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _CONFIRM_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Generate a confirmation for this order:\n{order_summary}"
                ),
            },
        ],
        temperature=0.3,
        seed=seed,
    )

    return response.choices[0].message.content.strip()


# -- Stage 4: LLM Judge ----------------------------------------------------

_JUDGE_SYSTEM = """\
You are a quality-assurance judge for a restaurant ordering system. You will \
receive an order JSON and a proposed confirmation text. Score how accurately \
the text reflects the order.

Check:
- Are all ordered items mentioned (name, size, quantity)?
- Are modifiers mentioned?
- Is the grand total correct and stated with a dollar sign?
- If a coupon was applied, is it mentioned?
- Does it end with a confirmation question?

Return JSON: {"confidence": 0.0-1.0, "issues": ["list of problems found"]}
A score of 1.0 means perfectly accurate. Deduct for each error.
"""


def _judge_confirmation(
    order: OrderDecision,
    confirmation_text: str,
    seed: int,
) -> JudgeResult:
    """Stage 4: Use LLM to score confirmation accuracy."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Order:\n{order.model_dump_json(indent=2)}\n\n"
                    f"Confirmation text:\n{confirmation_text}\n\n"
                    "Score this confirmation. Return JSON."
                ),
            },
        ],
        response_format={"type": "json_object"},
        temperature=0,
        seed=seed,
    )

    raw = response.choices[0].message.content
    data = json.loads(raw)
    return JudgeResult.model_validate(data)


def _fallback_confirmation(order: OrderDecision) -> str:
    """Deterministic fallback if judge rejects the LLM confirmation."""
    parts = []
    for item in order.normalizedOrder:
        desc = f"{item.qty}x {item.size + ' ' if item.size else ''}{item.name}"
        if item.modifiers:
            mod_strs = [m.name for m in item.modifiers]
            desc += f" ({', '.join(mod_strs)})"
        parts.append(desc)

    items_str = ", ".join(parts)
    line = f"Order: {items_str}."

    if order.total.couponApplied:
        line += (
            f" Coupon {order.total.couponApplied.code} applied"
            f" (-${order.total.couponApplied.discountAmount:.2f})."
        )

    line += f" Total: ${order.total.grandTotal:.2f}. Is that correct?"
    return line


# -- Orchestrator -----------------------------------------------------------

def validate_order(
    menu: Menu,
    order_text_or_json: Union[str, dict],
    tax_rate: float = DEFAULT_TAX_RATE,
    seed: int = DEFAULT_SEED,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> OrderDecision:
    """Validate a customer order against the normalized menu.

    4-stage pipeline:
      1. LLM extracts items/modifiers/coupon from customer text
      2. Python validates against menu and computes totals
      3. LLM generates friendly confirmation text
      4. LLM judge gates the confirmation (fallback if low confidence)
    """
    # Normalize input
    if isinstance(order_text_or_json, dict):
        order_input = json.dumps(order_text_or_json)
    else:
        order_input = order_text_or_json

    # Stage 1: LLM Extract
    extraction = _extract_order(menu, order_input, seed)

    # Stage 2: Python Validate & Calculate
    order = _calculate_order(menu, extraction, tax_rate)

    # Stage 3: LLM Confirm
    confirmation = _generate_confirmation(order, seed)

    # Stage 4: LLM Judge
    judge = _judge_confirmation(order, confirmation, seed)

    if judge.confidence >= confidence_threshold:
        order.confirmationText = confirmation
    else:
        order.confirmationText = _fallback_confirmation(order)

    return order


# ---------------------------------------------------------------------------
# Quick smoke-test (run directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_menu_text = """
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

    print("=== Parsing menu ===")
    menu = parse_menu(sample_menu_text)
    menu_json = menu.model_dump_json(indent=2)
    print(menu_json)

    with open("menu.json", "w") as f:
        f.write(menu_json)
    print("Saved to menu.json")

    if menu.priceWarnings:
        print("\n=== Price Warnings ===")
        for w in menu.priceWarnings:
            print(f"  WARNING: {w.message}")

    print("\n=== Validating order ===")
    order_text = (
        "I'd like a medium cheeseburger on a gluten-free bun with no pickles, "
        "and can I use coupon SPRING10?"
    )
    decision = validate_order(menu, order_text)
    order_json = decision.model_dump_json(indent=2)
    print(order_json)

    with open("order.json", "w") as f:
        f.write(order_json)
    print("Saved to order.json")
