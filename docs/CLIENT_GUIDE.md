# MaverickMCP Investment Analysis System

An evidence-based investment analysis system for Claude Desktop. Presents financial data objectively and interprets it through the lens of value investing principles.

---

## Philosophy

### Evidence First, Then Interpretation

MaverickMCP presents evidence objectively and applies value investing principles as an interpretive framework—not as rigid rules. The goal is informed decision-making, not dictated conclusions.

**For every analysis, you'll see:**
- **The Evidence** — Objective financial data, metrics, observable facts
- **Bull Case** — What supports the investment thesis
- **Bear Case** — What undermines it or creates risk
- **Value Investing Lens** — How Buffett's principles apply
- **Unanswered Questions** — What we don't know that matters
- **Assessment** — Where the evidence points, with appropriate nuance

When evidence is genuinely mixed, we'll say so. When reasonable people could disagree, we'll acknowledge it.

### The Value Investing Framework

These principles from Warren Buffett guide interpretation, but don't dictate conclusions:

| Principle | The Question It Asks |
|-----------|---------------------|
| **Circle of Competence** | Can we truly understand this business? |
| **Durable Moat** | What evidence supports lasting competitive advantage? |
| **Management Quality** | What does capital allocation history reveal? |
| **Margin of Safety** | What's the gap between price and intrinsic value? |
| **Long-term Ownership** | Would this make sense to own for a decade? |

### Assessment Spectrum

Instead of binary BUY/AVOID, analyses use a spectrum that reflects evidence clarity:

```
ASSESSMENT SPECTRUM
├── Strong conviction favorable (rare — evidence clearly aligns)
├── Favorable lean (most evidence positive, some uncertainty)
├── Mixed/Neutral (genuine uncertainty, reasonable arguments both ways)
├── Unfavorable lean (most evidence negative, some positive)
└── Strong conviction unfavorable (rare — evidence clearly aligns negatively)
```

**Conviction level** reflects how clear the evidence is, not how strongly we feel.

---

## Essential Tools

MaverickMCP exposes 10 essential tools for disciplined investment analysis:

| Tool | Purpose |
|------|---------|
| `comprehensive_stock_analysis` | Full parallel analysis: technical, risk, quant, valuation, simulation |
| `technical_analysis` | Price action, support/resistance, momentum indicators |
| `stock_screener` | Filter by fundamental and technical criteria |
| `risk_analysis` | VaR, CVaR, drawdown, stress testing, position sizing |
| `openbb_get_historical` | Price data across equities, crypto, forex, futures |
| `openbb_get_equity_quote` | Real-time quotes and basic metrics |
| `openbb_get_economic_indicator` | CPI, GDP, unemployment, interest rates, FRED data |
| `portfolio_manage` | Track positions with cost basis |
| `portfolio_analyze` | Portfolio risk, correlation, factor exposure |
| `macro_analysis` | Yield curve, fed funds, market regime, economic cycle |

*40+ additional tools remain callable by name when deeper analysis is needed.*

---

## Analysis Workflow

### Step 1: Establish Context

Before analyzing individual stocks, understand the environment:

```
"What's the current macro environment?"
"Where are we in the economic cycle?"
"What's the yield curve signaling?"
```

**Context matters because:** Great businesses can be poor investments in hostile environments, and mediocre businesses can outperform in favorable ones.

### Step 2: Gather Evidence

Run comprehensive analysis to collect objective data:

```
"Run comprehensive stock analysis on AAPL"
"Show me the financial metrics for MSFT"
"What's the valuation picture for JNJ?"
```

### Step 3: Review Both Sides

Every analysis should surface arguments for AND against:

```
"What's the bull case for NVDA?"
"What are the risks and bear arguments for GOOGL?"
"Where could the thesis go wrong?"
```

### Step 4: Apply the Framework

Use value investing principles as an interpretive lens:

```
"How durable is COST's competitive moat?"
"What margin of safety exists at current prices?"
"Is this within our circle of competence to evaluate?"
```

### Step 5: Assess with Nuance

Reach a conclusion that reflects the evidence clarity:

```
"Given the evidence, where does this fall on the assessment spectrum?"
"What conviction level is appropriate given the uncertainties?"
"What would need to change to shift the assessment?"
```

---

## Analysis Output Format

### Standard Analysis Structure

```
## [TICKER] Analysis

### The Evidence

**Financial Health**
- Revenue trend: [data]
- Profit margins: [data]
- Balance sheet: [data]
- Cash flow: [data]

**Competitive Position**
- Market share: [data]
- Industry dynamics: [observable facts]
- Competitive threats: [evidence]

**Valuation Metrics**
- Current P/E: [X] vs 5-year avg: [Y]
- P/FCF: [X]
- EV/EBITDA: [X]
- Dividend yield: [X%]

**Macro Context**
- Sector positioning in current cycle
- Interest rate sensitivity
- Geopolitical exposure

### Bull Case
- [Evidence-based point 1]
- [Evidence-based point 2]
- [What needs to go right for thesis to work]

### Bear Case
- [Evidence-based concern 1]
- [Evidence-based concern 2]
- [What could go wrong]

### Value Investing Lens

**Circle of Competence:** [How well can we understand this business?]

**Moat Assessment:** [What evidence supports/undermines competitive advantage?]

**Management Quality:** [What does capital allocation history suggest?]

**Intrinsic Value Range:** $[low] - $[high]
- Conservative case: [assumptions]
- Base case: [assumptions]
- Optimistic case: [assumptions]

**Margin of Safety:** [X%] at current price of $[Y]

### Unanswered Questions
- [Key uncertainty 1 — what we don't know that matters]
- [Key uncertainty 2]

### Assessment

**Position on spectrum:** [Favorable lean / Mixed / Unfavorable lean / etc.]

**Reasoning:** [Why the evidence points this direction]

**Conviction level:** [High/Medium/Low] — based on evidence clarity

**What would change this:**
- Upgrade triggers: [specific conditions]
- Downgrade triggers: [specific conditions]

**Note:** [Acknowledgment of where reasonable people might disagree]
```

---

## Example Analyses

### Example 1: Clear Favorable Case

```
## COST (Costco) Analysis

### The Evidence

**Financial Health**
- 15 consecutive years of revenue growth
- Operating margins stable at 3.4% (intentionally low — membership model)
- Net debt negative (more cash than debt)
- FCF conversion consistently strong

**Competitive Position**
- 90%+ membership renewal rate (extremely sticky)
- Largest warehouse club globally
- Buying power advantage compounds with scale

**Valuation**
- P/E: 52x (elevated vs market, but typical for COST)
- 5-year avg P/E: 40x
- Premium reflects quality, but historically reverts

### Bull Case
- Membership model creates recurring, predictable revenue
- 90%+ renewal rate is exceptional customer loyalty evidence
- International expansion runway (underpenetrated in Asia, Europe)
- Inflation-resistant value proposition strengthens in uncertain times
- Management conservative, shareholder-aligned (special dividends)

### Bear Case
- 52x P/E leaves little room for disappointment
- Same-store sales growth moderating from pandemic highs
- Amazon/Walmart competitive pressure on general merchandise
- Labor costs rising, may pressure margins
- Premium valuation assumes continued execution

### Value Investing Lens

**Circle of Competence:** High — retail business model is straightforward.
Pay annual fee, get access to bulk goods at thin margins. Economics are clear.

**Moat Assessment:** Strong evidence of durability
- Membership renewal rates (90%+) demonstrate switching costs
- Scale advantages in purchasing difficult to replicate
- Real estate portfolio (owned, not leased) is underappreciated asset

**Management Quality:** Excellent track record
- Founder-led culture maintained post-transition
- Capital allocation disciplined (special dividends vs empire building)
- Employee treatment creates operational advantage

**Intrinsic Value Range:** $750 - $950
- Conservative (35x normalized earnings): $750
- Base (42x reflecting quality): $850
- Optimistic (50x premium sustained): $950

**Margin of Safety:** Limited at $920 (current price at high end of range)

### Unanswered Questions
- Will premium multiple compress if growth slows further?
- How much international expansion is already priced in?
- Can membership fee increases continue without churn?

### Assessment

**Position on spectrum:** Favorable lean

**Reasoning:** Business quality is exceptional with clear, durable competitive
advantages. Management track record is strong. However, valuation provides
limited margin of safety — you're paying a fair price for a great business,
not a bargain price.

**Conviction level:** Medium — business quality high conviction,
but entry valuation introduces uncertainty about forward returns

**What would change this:**
- Upgrade: Price decline to $750 range (creates margin of safety)
- Downgrade: Membership renewal decline below 88%, margin compression

**Note:** Growth-oriented investors comfortable with quality premiums may view
this more favorably. Deep value investors seeking margin of safety will likely
wait for better entry points.
```

### Example 2: Genuinely Mixed Case

```
## TSLA (Tesla) Analysis

### The Evidence

**Financial Health**
- Revenue CAGR 50%+ over 5 years (exceptional growth)
- Automotive margins compressed from 25% to 17% (price cuts)
- $26B cash, $2B debt (fortress balance sheet)
- FCF positive but volatile with capex cycles

**Competitive Position**
- EV market share declining as competition intensifies
- Brand strength remains, especially in US market
- Manufacturing efficiency (Austin, Berlin) improving
- Charging network is genuine competitive advantage

**Valuation**
- Forward P/E: 65x
- P/S: 8x (vs auto industry <1x)
- Priced for significant non-auto revenue (energy, FSD)

### Bull Case
- Global EV transition creates multi-decade tailwind
- Energy storage business growing 100%+ (underappreciated)
- FSD technology, if realized, transforms unit economics
- Manufacturing expertise and cost position improving
- Musk's execution track record despite controversy
- Brand loyalty creates pricing power

### Bear Case
- 65x forward P/E requires exceptional execution across multiple bets
- FSD timeline repeatedly missed — regulatory path unclear
- Chinese EV competition intensifying (BYD, NIO, Xiaomi)
- Legacy automakers scaling EV production (Ford, GM, VW)
- Margin pressure may be structural, not cyclical
- Key person risk (Musk distraction, reputation)
- Regulatory tailwinds (EV credits) diminishing

### Value Investing Lens

**Circle of Competence:** Challenging
The bull case requires assessing: (1) autonomous driving technology timelines,
(2) robotics potential, (3) energy storage market evolution. These are difficult
to evaluate with confidence.

**Moat Assessment:** Mixed evidence
- Brand and charging network: durable advantages with evidence
- Manufacturing: improving but replicable
- Technology: unproven at scale, timeline uncertain

**Management Quality:** Complicated
- Execution on production scaling: exceptional
- Capital allocation: reasonable (no dilution, minimal debt)
- Focus and attention: legitimately uncertain given other ventures

**Intrinsic Value Range:** Wide dispersion reflecting uncertainty
- Conservative (auto-only, compressed multiples): $120
- Base (moderate energy/FSD value): $250
- Optimistic (significant FSD/robotaxi value): $500+

**Margin of Safety:** Negative at $380 vs conservative case

### Unanswered Questions
- When (if ever) does FSD achieve regulatory approval for robotaxi?
- Will energy storage scale profitably?
- Is margin compression cyclical or structural?
- How much automotive market share loss is acceptable?

### Assessment

**Position on spectrum:** Mixed/Neutral for traditional value investors

**Reasoning:** The evidence genuinely supports both bullish and bearish
conclusions depending on which assumptions you make about unknowable future
outcomes. The business has real strengths, but the valuation requires success
in areas we cannot confidently assess.

**Conviction level:** Low — not because the business is bad, but because
the thesis depends on outcomes outside our analytical confidence

**What would change this:**
- More favorable: Price decline creating margin of safety ($150-180 range),
  or concrete FSD regulatory approval
- Less favorable: Continued margin compression, market share erosion
  accelerating beyond EV transition

**Note:** This is a case where reasonable, intelligent investors can reach
opposite conclusions. Growth investors with tech expertise and higher risk
tolerance may see compelling opportunity. Traditional value investors will
likely pass due to valuation and circle of competence concerns. Neither
view is obviously wrong.
```

### Example 3: Clear Unfavorable Case

```
## [Hypothetical Overleveraged Company] Analysis

### The Evidence

**Financial Health**
- Revenue declining 8% annually for 3 years
- Operating margins negative (-5%)
- Debt/EBITDA: 8x (dangerous territory)
- Negative free cash flow, burning cash

**Competitive Position**
- Market share eroding to lower-cost competitors
- Product relevance declining (secular headwind)
- No clear path to competitive differentiation

### Bull Case
- Asset value exceeds market cap (liquidation floor)
- Turnaround management recently hired
- Industry consolidation could drive acquisition premium

### Bear Case
- Leverage makes equity highly vulnerable
- Declining business cannot service debt long-term
- Turnarounds in secular decline rarely succeed
- Time is enemy with cash burn

### Value Investing Lens

**Circle of Competence:** Business is understandable, but thesis requires
predicting turnaround success — historically low probability

**Moat Assessment:** No evidence of competitive advantage; business is
commodity with cost disadvantage

**Management Quality:** New team, no track record to evaluate

**Margin of Safety:** Apparent cheapness is likely value trap — assets
may not realize stated value in distress

### Assessment

**Position on spectrum:** Strong conviction unfavorable

**Reasoning:** Declining business with dangerous leverage and no competitive
advantage. Apparent cheapness reflects genuine distress, not opportunity.
Asset floor less reliable with forced liquidation potential.

**Conviction level:** High — evidence clearly points one direction

**What would change this:**
- Debt restructuring that materially de-risks equity
- Concrete evidence of business stabilization (not promises)

**Note:** Distressed debt specialists may find opportunity in the capital
structure, but equity appears to carry uncompensated risk.
```

---

## Intellectual Honesty Principles

### What We Acknowledge

**Limitations we're explicit about:**
- Valuation is estimation, not precision — we provide ranges, not point targets
- The future is uncertain — macro conditions, competitive dynamics, and management decisions can change
- Our circle of competence has boundaries — some businesses are harder to evaluate
- Timing is beyond our ability — we assess value, not when the market will recognize it

**When we say "we don't know":**
- Technology trajectories requiring specialized expertise
- Regulatory outcomes dependent on political processes
- Consumer behavior shifts without historical precedent
- Short-term price movements

### Questions We Always Ask

1. **What's the evidence?** (Not: what do we want to believe?)
2. **What are we assuming?** (And are those assumptions testable?)
3. **What would prove us wrong?** (And are we watching for it?)
4. **Where might we be biased?** (Recent performance, narrative appeal, etc.)
5. **Would Buffett understand this?** (Circle of competence check)

---

## Portfolio Management

### Tracking Positions

```
"Add 100 shares of AAPL at $175 to my portfolio"
"Show my portfolio with current P&L"
"What's my overall risk exposure?"
```

### Reviewing Holdings

For each position, periodically reassess:
- Has the original thesis changed?
- Is the evidence still supportive?
- Has the margin of safety expanded or compressed?
- Are there better opportunities for the capital?

### Position Sizing Philosophy

- **Higher conviction + clearer evidence** → larger position (but diversified)
- **Mixed evidence or lower conviction** → smaller position or avoid
- **Correlation awareness** — don't concentrate in correlated bets

---

## Macro Analysis

### Understanding Context

```
"What phase of the economic cycle are we in?"
"How is the yield curve positioned?"
"What are the dominant geopolitical risks?"
```

### Why Macro Matters

Great businesses in hostile macro environments can underperform. The context doesn't override fundamental analysis, but it influences:
- Sector positioning
- Valuation multiples market will pay
- Risk appetite appropriate for current conditions

---

## Key Reminders

### Before Analysis

1. **Seek evidence, not confirmation** — Actively look for disconfirming data
2. **Acknowledge uncertainty** — Comfort with "I don't know" is a feature
3. **Remember the framework** — Buffett's principles are guides, not laws

### During Analysis

1. **Present both sides** — Bull and bear cases deserve fair treatment
2. **Be specific about assumptions** — What must be true for thesis to work?
3. **Quantify when possible** — Ranges > point estimates > hand-waving

### After Analysis

1. **Match conviction to evidence clarity** — Don't overstate confidence
2. **Acknowledge reasonable disagreement** — Smart people can see it differently
3. **Define what would change the view** — Make it falsifiable

---

## Quick Reference

### Essential Commands

| Need | Command |
|------|---------|
| Full analysis | "Run comprehensive analysis on [TICKER]" |
| Macro context | "What's the current macro environment?" |
| Valuation | "What's the intrinsic value range for [TICKER]?" |
| Both sides | "What are the bull and bear cases for [TICKER]?" |
| Risk assessment | "Run risk analysis on [TICKER]" |
| Portfolio review | "Analyze my portfolio" |

### The Buffett Checklist

- [ ] Do I understand this business?
- [ ] Is there evidence of a durable moat?
- [ ] Is management aligned with shareholders?
- [ ] Does margin of safety exist at current price?
- [ ] Would I be comfortable owning this for 10 years?
- [ ] What could go wrong?

---

*MaverickMCP — Evidence-based analysis through a value investing lens*

*"Risk comes from not knowing what you're doing."* — Warren Buffett
