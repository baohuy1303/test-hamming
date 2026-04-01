# Backend Coding Challenge: Menu Parser
https://hammingai.notion.site/Backend-Coding-Challenge-Menu-Parser-2981f2fbf30b8016a549f65f4d2b9d49 

Build a service that converts free-form menu text into a structured menu and then validates customer orders against it, proposing fixes and upsells when appropriate. This tests **structured prompting**, **schema design**, and **rule checking**.

---

## Task

Implement two functions:

```python
parse_menu(menu_text) -> Menu
validate_order(menu: Menu, order_text_or_json) -> OrderDecision
```

- **`parse_menu`** turns messy menu copy (categories, items, sizes, modifiers, prices, taxes) into a clean JSON schema.
- **`validate_order`** takes either a raw customer order (text) or a partial JSON order and returns:
  - `normalizedOrder` — items, sizes, modifiers, quantities, price breakdown
  - `issues` — e.g., `"gluten-free bun not available on kids menu"`
  - `suggestions` — e.g., `"upgrade to large drink for $0.50"`
  - `total` — with tax, plus a short confirmation string the agent could read back

---

## Inputs

| Parameter | Description |
|---|---|
| `menu_text` | A few paragraphs of unstructured text (copy-pasted menu, typos allowed) |
| `order_text_or_json` | Multi-item orders, customizations, removals, coupons — as raw text or partial JSON |

---

## Output Shape

```json
{
  "normalizedOrder": [
    {
      "sku": "CB-MED",
      "name": "Cheeseburger",
      "size": "Medium",
      "modifiers": ["no pickles", "gluten-free bun"],
      "qty": 1,
      "unitPrice": 8.99
    }
  ],
  "issues": [
    "Gluten-free bun adds $1.00",
    "Coupon SPRING10 is invalid on combo items"
  ],
  "suggestions": [
    "Upgrade drink to large for +$0.50",
    "Add fries to make it a combo and save $1.00"
  ],
  "total": {
    "subtotal": 9.99,
    "tax": 0.82,
    "grandTotal": 10.81
  },
  "confirmationText": "One medium cheeseburger on a gluten-free bun, no pickles. Total is $10.81. Is that correct?"
}
```

---

## Constraints

- **Noise robustness** — handle spelling errors, inconsistent spacing, and malformed prices gracefully
- **Determinism** — support a seed/temperature option for consistent parsing decisions across runs
- **Business rules** — enforce availability windows, modifier compatibility, and coupon scope
- **Numeric sanity** — correct rounding behavior; accept tax rate as input or fall back to a sensible default

---

## Bonus Challenges

### Price Validator
Flag menu inconsistencies and emit a diff:
> `"Large Soda listed as $2.49 and $2.59 — pick one"`

### Multi-language Support
Accept menu snippets in multiple languages and normalize them into a single consistent schema regardless of source language.

### Coupon / Discount Engine
Implement stack rules for promotions:
> `"Combo discount excludes BOGO"`

### LLM-as-Judge Read-back Check
Add auto-verification hints that confirm a voice agent's spoken read-back matches `normalizedOrder` — useful for QA in voice ordering pipelines.

### Modifier DSL
Define a tiny domain-specific language for modifiers with:
- `required` / `optional` flags
- Cardinality constraints (e.g., `"pick 1–2"`)
- Conflict declarations (e.g., `"gluten-free bun conflicts with brioche bun"`)
- Automatic repair suggestions when an order violates the rules

---

## Notes

- Use **LLMs** to solve the core parsing and validation logic.
- The schema should be expressive enough to represent real-world menus with nested categories, size tiers, and conditional modifiers.
