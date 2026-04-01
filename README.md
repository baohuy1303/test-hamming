# Menu Parser

## Architecture and Design Decisions

This was a fun challenge. It took me a lot more time than I expected once I started doing the coupon stacking and DSL for modifiers, had to spend some time considering cases that the customer and menu owner might put in to design my schema. But here's my final design for a menu parser and order validator.

The challenge asks for two functions: `parse_menu` and `validate_order`. Naive or brute-force approach would be to just call 2 LLMs and let it handle everything but: the LLM miscalculates tax, hallucinates once it needs to reason all the coupon rules, DSL etc... it starts inventing prices, drops coupons, and produces different totals on repeated runs.

The core design decision was to split the work between AI generation and deterministic Python code. This led to a **4-stage pipeline** for `validate_order`:

1. **LLM Extract** (gpt-4o-mini) -- Parse the customer's natural language into structured items, modifiers, and coupon codes.
2. **Python Validate and Calculate** -- Match extracted items against the menu by SKU. Validate modifiers against the DSL rules. Run the coupon stacking engine. Compute subtotal, discount, tax, and grand total in Python with explicit rounding.
3. **LLM Employee** (gpt-4o-mini) -- Generate a friendly read-back confirmation from the validated order data.
4. **LLM Judge** (gpt-4o-mini) -- Score the confirmation text against the order JSON. If confidence falls below 0.85, substitute a deterministic fallback template.

`parse_menu` is a single gpt-4o call with a detailed system prompt and all the JSON schemas. A post-parse price validator catches inconsistencies the LLM might introduce.

### What was built beyond the base requirements for bonuses

**Modifier DSL.** Menu items have `modifierGroups` instead of flat modifier lists. Each group defines:
- `required` / `optional` flag
- `minSelections` / `maxSelections` cardinality (e.g., "pick exactly 1 bun")
- Per-option `conflicts` list (e.g., "gluten-free bun conflicts with pretzel bun")

Stage 2 validates customer selections against these rules and produces both issues ("gluten-free bun conflicts with pretzel bun") and repair suggestions ("Remove either gluten-free bun or pretzel bun"). Surcharges always come from the menu definition, never from the customer's input.

**Coupon stacking engine.** Coupons have structured fields extracted by the LLM: `stackable`, `excludes` (other coupons it cannot combine with), `appliesTo` (eligible items), and `excludesItems` (blocked items). The Python engine processes multiple coupons in order, checking stackability, mutual exclusions, and item-level restrictions before computing each discount. Three discount types are supported:
- `percent` -- percentage off eligible subtotal
- `flat` -- fixed dollar amount off
- `freeItem` -- grants a menu item at $0.00 (e.g., "Free small fries with any burger purchase"), with prerequisite checking via `appliesTo`
This is just a medium and more simplified version of coupon stacking, and it doesn't cover all the most complex cases, but it should work fine for most common cases.

Coupon outcomes (applied, rejected with reason, or granted free item) are in both the LLM confirmation and the deterministic fallback.

**Price validator.** After parsing, `_validate_menu_prices` groups items by (name, size) and flags conflicting prices or zero/negative prices.

**Multi-language support.** `parse_menu` accepts a `translate=True` flag. When set, a lightweight gpt-4o-mini call translates the menu text to English before parsing, preserving prices and structure for simpler processing.

**LLM-as-judge.** Stage 4 scores the LLM-generated confirmation against the order data. If the confidence score is below the threshold, a deterministic template replaces the LLM output. This catches hallucinated totals or dropped items without requiring a human reviewer.

**Combo support.** Menu items can declare `comboIncludes` (list of included item names) and `notes` (e.g., "coupon codes not valid on combo items").

## How to Run and Test

Requirements: Python 3.11+, an OpenAI API key.

```bash
# Setup
python -m venv venv
source venv/Scripts/activate   # Windows
pip install -r requirements.txt

# Set your API key
echo "OPENAI_API_KEY=sk-..." > .env

# Run the smoke test (parses sample menu, validates a sample order, writes menu.json and order.json)
python main.py

# Run the full test suite (49 tests)
pytest test_main.py -v
```

The test suite has 6 test classes. `TestPriceValidator`, `TestModifierDSL`, and `TestCouponStacking` are pure Python (no API calls, run instantly). `TestParseMenu`, `TestValidateOrder`, and `TestMultiLanguage` call the OpenAI API.
