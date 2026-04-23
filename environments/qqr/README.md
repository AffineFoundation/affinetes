# QQR Travel Planning Evaluation Environment

## 1. Environment Value

### 1.1 Why QQR

Mainstream benchmarks (MMLU, HumanEval, GSM8K, etc.) evaluate **static knowledge** or **closed-form reasoning**. But real-world LLM applications increasingly involve acting as **Agents** — autonomously calling external tools, integrating multi-source information, and generating structured outputs.

QQR fills this gap: it evaluates **end-to-end agent capability for completing complex open-ended tasks with tools**.

Specifically, QQR uses "Chinese travel planning" as the evaluation vehicle, requiring the model under test to:
1. **Understand** natural language travel requirements (multiple types, constraints, and preferences)
2. **Autonomously decide** which MCP tools to call, in what order, and with what parameters
3. **Integrate** results from multiple tools (POI info, navigation data, weather forecasts, flight/train queries)
4. **Generate** a structured travel plan that is factually accurate, complete, and logically coherent

### 1.2 Core Evaluation Capabilities

| Capability | Evaluation Method | Why It Matters |
|------------|------------------|----------------|
| **Tool selection** | Whether the required tool set was called | Agents need to know "which tool to use" |
| **Tool usage quality** | Whether parameters are correct (coordinate format, date format, etc.) | Calling a tool with wrong parameters = wasted call |
| **Information extraction & integration** | How much real tool data is cited in the output | Distinguishes "actually using tools" from "called but ignored results" |
| **Content completeness** | Whether all necessary planning dimensions are covered | Good plans need transportation, lodging, dining, budget, etc. |
| **Factual accuracy** | Whether flight numbers, train IDs, prices are traceable to tool results | Detects hallucination |
| **Output quality** | LLM-judged practicality, analysis depth, logic, user experience, factual grounding | Automated proxy for human preference |

### 1.3 Value for RL Training

```
Episode flow:
  reset(task_id) → Generate problem + initial prompt
       ↓
  step(tool_calls) → Execute tools → Return step_reward (0~1)
       ↓  (loop ≤ 15 steps)
  step(final_answer) → Final scoring → Return final_reward (0~100)
```

- **Step rewards**: Immediate reward per step (0.4×tool_selection + 0.3×argument_quality + 0.3×result_usefulness), guiding tool-calling policy learning
- **Deterministic scoring**: Same task_id + same epoch salt = identical transport data and scores, reproducible training
- **Smooth LLM-code coupling**: `llm_score *= min(1.0, code / (50 × 0.6))`, prevents optimizing only one dimension
- **10K+ task space**: 7 types × 3 difficulties × 70+ cities × weekly salt rotation, prevents overfitting

---

## 2. System Architecture

### 2.1 Overall Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     QQR Evaluation Flow                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  problem_generator.py + knowledge_graph.py                      │
│  ┌─────────────┐     task_id (deterministic seed)               │
│  │ 7 problem    │ ──→ TravelProblem struct ──→ Chinese prompt   │
│  │   types      │     (cities/dates/budget/preferences/         │
│  │ 3 difficulty │      constraints)                             │
│  │   levels     │     City knowledge graph provides             │
│  │ 70+ cities   │     seasons/specialties/landmarks             │
│  └─────────────┘                                                │
│         ↓                                                       │
│  env.py (Actor) — Two-phase design                              │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Phase 1: Tool-calling Loop (≤ 15 steps)               │    │
│  │                                                         │    │
│  │  Model under test ←→ MCP Tool Set (via MCPState)        │    │
│  │              ├── poi_search      (AMap API, real)        │    │
│  │              ├── around_search   (AMap API, real)        │    │
│  │              ├── direction       (AMap API, real)        │    │
│  │              ├── weather         (AMap API, real)        │    │
│  │              ├── search_flights  (deterministic, mock)   │    │
│  │              └── search_train    (deterministic, mock)   │    │
│  │                                                         │    │
│  │  Per step → StepRewardCalculator → step_reward (0~1)    │    │
│  │                                                         │    │
│  │  Phase 2: Final Answer                                  │    │
│  │  If model's natural answer is insufficient →            │    │
│  │  send final-answer prompt (tools=None, no more calls)   │    │
│  └─────────────────────────────────────────────────────────┘    │
│         ↓ (model outputs final plan)                            │
│  scorer.py + llm_validator.py                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Multi-layer Scoring (max 100, 50/50 code-LLM split)    │    │
│  │                                                         │    │
│  │  1. Code Score (always computed first, 50 pts)          │    │
│  │     tool_info_used ──→ Pure code gate (IC-only)          │    │
│  │     info_consistency (25) ← 10-category fact comparison  │    │
│  │     completeness (25) ← proximity-based grounding        │    │
│  │     fabrication_penalty (0 ~ -12.5) ← hallucination      │    │
│  │                                                         │    │
│  │  2. Hard Constraints (threshold checks)                 │    │
│  │     format_valid ──→ fail = ×0.15 (near-zero, RL grad)  │    │
│  │     tool_info_used ──→ fail = 0 (code-determined)        │    │
│  │     required_tools ──→ fail = ×0.5                       │    │
│  │     poi_names ──→ fail = ×0.7                            │    │
│  │     transport_grounded ──→ fail = ×0.3 (progressive)     │    │
│  │     tool_quality ──→ fail = ×0.5                         │    │
│  │                                                         │    │
│  │  3. LLM Score (optional enhancement, 50 pts)            │    │
│  │     UnifiedScorer: single call for all 5 dimensions      │    │
│  │     practicality + analysis_depth + logic                │    │
│  │     + user_experience + factual_grounding                │    │
│  │     × smooth coupling with code score                    │    │
│  │     × defense-in-depth (fg<4 → compress IC/Comp)         │    │
│  │                                                         │    │
│  │  Final = (code + llm) × HC_multipliers                   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 File Structure

```
environments/qqr/
├── __init__.py             # Package exports: Actor, ProblemGenerator, TravelScorer, etc.
├── env.py                  # Actor class: two-phase Agent Loop, MCP tool dispatch, evaluate() entry
├── scorer.py               # Core scoring: fact extraction, HC checks, IC/Comp, fabrication detection
├── config.py               # Config: tool definitions, city lists, score weights, HC penalties, transport cost floors, city-pair assertions
├── problem_generator.py    # Deterministic problem generator: 7 types, DifficultyProfile, prompt templates
├── knowledge_graph.py      # City knowledge graph: 71 cities with specialties/landmarks/food themes/seasons/transport hubs
├── parser.py               # Output parser: JSON-first + regex fallback structured extraction
├── llm_validator.py        # LLM semantic evaluation: UnifiedScorer 5 dims × 10 pts, structured summary + anti-injection/anti-echo
├── mcp_wrapper.py          # MCP protocol wrapper (ported from QQR, avoids slime dependency)
├── mock_transport/
│   ├── __init__.py
│   └── server.py           # Deterministic transport data generation (SHA256 seed, 70+ cities, epoch salt)
├── Dockerfile              # Container build config
└── requirements.txt        # Dependencies
```

### 2.3 MCP Tool Set

| Tool | Data Source | Purpose | Returns |
|------|------------|---------|---------|
| `poi_search` | AMap API (real) | Search POIs (attractions/hotels/restaurants) | Name, address, coordinates, rating, phone |
| `around_search` | AMap API (real) | Radius-based nearby search | Nearby POI list |
| `direction` | AMap API (real) | Route planning (driving/walking/cycling/transit) | Distance, duration, route description |
| `weather` | AMap API (real) | Weather forecast | Conditions, temperature, wind |
| `search_flights` | Deterministic (mock) | Flight search | Flight number, price, time, airline |
| `search_train_tickets` | Deterministic (mock) | Train ticket search | Train number, price, time, seat type |

**Note**: All tools only support domestic Chinese cities (71 cities). International destinations are not supported. AMap API covers mainland China only.

**Why mock transport data?** Real flight/train APIs are unstable and rate-limited. mock_transport generates deterministic data via SHA256 seeds, guaranteeing:
- Same `(date, from_city, to_city, salt)` → identical flight/train data
- Weekly `TRANSPORT_SALT` rotation → prevents models from memorizing historical data
- 70+ city interconnections → covers short/medium/long-haul scenarios

**AMap Cache Epoch Alignment**: AMap data (POI, weather, etc.) comes from real APIs and is time-varying. To ensure reproducible scoring within the same evaluation epoch (week), AMap's `cache_ttl` is aligned to the `TRANSPORT_SALT` weekly epoch boundary (`max(86400, epoch_end - now)`) rather than using a fixed TTL. This keeps AMap data stable within an epoch after the first query, so the `info_consistency` comparison baseline does not drift within a batch.

**Standard OpenAI Tool Calling**: Actor uses the fully standard OpenAI function calling format: assistant messages carry the `tool_calls` field (with `id`/`type`/`function`), and tool results are returned via `role: "tool"` + `tool_call_id`. The conversation history strictly follows the `assistant(tool_calls) → tool → assistant` state machine, ensuring models correctly distinguish reasoning / tool execution / tool result across multi-turn calls.

---

## 3. Scoring System Details

### 3.1 Total Score Formula & Execution Order

```
Total score formula:
  total = (code_coupled + llm_adjusted) × HC_multiplier

Where:
  code_total = max(0, 50 × sqrt(IC/25 × Comp/25) × diversity_mult + Fab)
             = geometric_mean(IC, Comp) × diversity_mult + fabrication_penalty
             Note: geometric mean penalizes IC/Comp imbalance (e.g., high IC + low Comp hack patterns)

  llm_raw    = practicality + analysis_depth + logic + user_experience + factual_grounding

  Bidirectional Coupling:
  1. LLM constrained by code:
     code_ratio = min(1.0, code_total / (50 × 0.6))
     llm_adjusted = llm_raw × code_ratio

  2. Code constrained by LLM (when LLM available):
     llm_ratio = min(1.0, llm_total / (50 × 0.4))
     code_coupled = code_total × (0.7 + 0.3 × llm_ratio)    ← code retains min 70%

  Monotonicity guarantee:
     base = max(code_coupled + llm_adjusted, 0.5 × (code_total + llm_raw))
     → coupling never drops total below 50% of raw sum

  HC_multiplier = ∏(penalty_i for each failed constraint)   ← all failed HCs multiply

Range:
  Code Score:  0 ~ 50   (geometric_mean(IC 25, Comp 25) × diversity + Fab 0~-12.5)
  LLM Score:   0 ~ 50   (5 dimensions × 10)
  Total:       0 ~ 100
```

**Algorithmic vs LLM Scoring — Overview**

| Score Component | Max | Method | Description |
|-----------------|-----|--------|-------------|
| **Hard Constraints** | | | |
| format_valid | ×0.15 | **Algorithmic** | Regex matches problem-type keywords + min length ≥ 200 chars |
| tool_info_used | ×0.0 / ×0.05 | **Algorithmic** (IC-only) | Transport types: IC≥6; Non-transport: IC≥8 (fail → ×0.05 softened) |
| required_tools_called | ×0.5 | **Algorithmic** | Coverage threshold + core tools + transport tool check |
| poi_names_verified | ×0.7 | **Algorithmic** | Fuzzy-match POI names ≥ 2 |
| transport_grounded | ×0.3~1.0 | **Algorithmic** | Set intersection to verify flight/train IDs, prices, times |
| tool_quality | ×0.5 | **Algorithmic** | coverage_ratio + validity_ratio ≥ 50% |
| **Code Score (50)** | | | |
| info_consistency | 25 | **Algorithmic** | 10-category fact extraction → set intersection/fuzzy match → ratio scoring + min quantity threshold |
| completeness | 25 | **Algorithmic** | Proximity-based tiered verification × quantity scaling (no free tier) |
| fabrication_penalty | 0~-12.5 | **Algorithmic** | Price error detection + weather fabrication + transport fabrication deduction |
| **LLM Score (50)** | | | |
| practicality | 10 | **LLM** | Plan feasibility (time coordination, transport rationality) |
| analysis_depth | 10 | **LLM** | Depth of analysis (penalizes data copying/echo, rewards reasoning and limitation awareness) |
| logic | 10 | **LLM** | Logical coherence (route planning, geographic grouping, penalizes planning based on fabricated POIs) |
| user_experience | 10 | **LLM** | User need satisfaction (constraint response, preference reflection) |
| factual_grounding | 10 | **LLM** | Factual accuracy (are flights/trains/prices/POIs traceable to tool data?) |

> **Summary**: Of 100 points, **50 are purely algorithmic** (deterministic, reproducible), **50 are LLM-judged** (semantic quality). `tool_info_used` is entirely code-determined (based on IC threshold, no Comp), independent of LLM. LLM scores are constrained by code scores via coupling (code < 30 → linear compression), ensuring high LLM scores must be backed by code scores. When LLM is unavailable, the maximum possible score is 50 (code only).

**Scoring Execution Order** (actual flow in `TravelScorer.score()` — code-first architecture):

```
1. Parse output → ParsedOutput (JSON-first + regex fallback)
2. Hard Constraint checks → format_valid, required_tools_called, poi_names_verified, transport_grounded
3. tool_quality gate → coverage_ratio < 0.5 OR validity_ratio < 0.5 → HC flag
4. Compute info_consistency (25 pts) ← with min quantity threshold + context-sensitive matching
5. Compute completeness (25 pts) ← proximity-based grounding (no free tier)
6. Pure code tool_info_used gate → IC≥6 (transport) or IC≥8 (non-transport), IC-only, no Comp
   └── tool_info_used=False → total=0 (transport) or ×0.05 (non-transport, softened)
7. Fabrication detection → fabrication_penalty (0 ~ -12.5)
8. LLM validation (optional enhancement) → 5-dimension scores (50 pts)
   ├── LLM available → fill practicality/analysis_depth/logic/ux/factual_grounding
   ├── Cross-validation: total > 36/50 → second model re-evaluates, take min
   └── LLM unavailable → code score only, log error (no zeroing)
9. Defense-in-depth: fg < 4/10 → compress IC and Comp (min(regex_mult, llm_grounding_mult))
10. Assemble ScoreBreakdown → .total property auto-computes final score
```

### 3.2 Hard Constraints (Threshold Checks)

All HCs are multiplicative penalties. Multiple failures **multiply together**. E.g., `required_tools_called`(0.5) + `poi_names_verified`(0.7) both fail → total × 0.35.

#### 3.2.1 format_valid (multiplier 0.15)

Checks whether the output contains **basic structure** of a travel plan. Uses different regexes per problem type:

| Problem Type | Check Condition |
|-------------|-----------------|
| intercity | Has transport options or matches `(航班\|火车\|高铁\|飞机\|车次)` |
| multiday | Has daily itinerary or matches `第N天` / `Day N` |
| hybrid | Has transport or daily itinerary |
| single_poi | Matches `(景点\|游览\|路线\|门票\|开放)` |
| food_tour | Matches `(美食\|餐厅\|小吃\|特色\|推荐)` |
| business | Matches `(航班\|火车\|高铁\|酒店\|商务)` |
| family_study | Matches `(亲子\|儿童\|学习\|博物馆\|科技馆\|体验)` |

Also requires output length ≥ 200 characters (`FORMAT_MIN_LENGTH`). Failure multiplier is 0.15 (not 0), preserving RL gradients.

#### 3.2.2 tool_info_used (IC-only Gate)

**Entirely code-determined**, independent of LLM. Based on epoch-salted fact overlap ratio (IC score only), cannot be faked:

```
Transport types (intercity/hybrid/business):
  IC ≥ 6.0 → tool_info_used = True
  Otherwise → tool_info_used = False → total = 0

Non-transport types (multiday/single_poi/food_tour/family_study):
  IC ≥ 8.0 → tool_info_used = True
  Otherwise → tool_info_used = False → total × 0.05 (softened, not hard zero)
```

**Why no Comp?** Comp measures content coverage quality, not tool usage — a model may cite all tool data (high IC) but miss certain content patterns (low Comp). Comp is already penalized via geometric mean; penalizing again through tool_info_used would be double-jeopardy.

Production data validation: genuine tool usage yields IC≈25, Comp≈25; fabrication/no-tool yields IC≈0, Comp≈0. Thresholds of 6-8 have sufficient safety margin.

#### 3.2.3 required_tools_called (multiplier 0.5)

Three-layer check:

1. **Coverage threshold**: `called ∩ required / |required|` must meet the per-type threshold:

| Problem Type | Threshold | required_tools |
|-------------|-----------|----------------|
| intercity | 60% | poi_search, direction, weather, flights, trains |
| multiday | 60% | poi_search, around_search, direction, weather |
| hybrid | 60% | All 6 tools |
| single_poi | 60% | poi_search, around_search, direction, weather |
| food_tour | 60% | poi_search, around_search, direction, weather |
| business | 60% | poi_search, direction, weather, flights, trains |
| family_study | 60% | poi_search, around_search, direction, weather |

2. **Core tools**: Tools in `CORE_TOOLS_BY_TYPE` must all be called. Most types have `{poi_search}` as the core tool; intercity has an empty set (since the 60% threshold + REQUIRES_TRANSPORT already suffice).

3. **Transport tools**: intercity/hybrid/business types must call at least one of `search_flights` or `search_train_tickets`.

#### 3.2.4 poi_names_verified (multiplier 0.7)

Checks whether at least **2** POI names in the output come from `poi_search`/`around_search` results. Uses three-tier matching:
1. Exact containment match
2. Normalized match (strip punctuation/spaces then containment)
3. Half-match (for names ≥ 4 characters, first or second half appearing counts)

If the model didn't call POI tools, or tools returned no POIs, this check auto-passes.

#### 3.2.5 transport_grounded (progressive multiplier 0.3 ~ 1.0)

**Only applies to** intercity/hybrid/business types. Verifies three types of transport claims:

| Verification | Method | Strictness |
|-------------|--------|------------|
| Transport IDs (flight/train numbers) | Set intersection: `output_ids ∩ tool_ids` | 100% must match |
| Transport prices (associated with IDs) | Price error ≤ 15% | 70% match rate |
| Transport times (associated with IDs) | Exact string match | 70% match rate |

**Progressive penalty** (not binary pass/fail):
```
fabrication_ratio = unverified_claims / total_transport_claims

if fab_ratio ≤ 0.2:   multiplier = 1.0      (no penalty)
if fab_ratio = 0.5:    multiplier ≈ 0.74
if fab_ratio = 1.0:    multiplier = 0.3      (max penalty)

Formula: multiplier = 1.0 - (1.0 - 0.3) × (fab_ratio - 0.2) / 0.8
```

**Special case**: If the model called transport tools but they returned empty/error, those transport claims are marked `unverifiable` and excluded from the fabrication ratio.

#### 3.2.6 tool_quality (multiplier 0.5)

Both metrics must be ≥ 50% to pass:
- **coverage_ratio** = `|called ∩ required| / |required|` (same as 3.2.3 coverage)
- **validity_ratio** = valid_calls / total_calls
  - Valid call = required params present + non-empty non-error result → 1.0
  - Params present but error result → 0.5
  - Missing params → 0

### 3.3 Info Consistency (Information Consistency, 25 pts)

Measures "how much information in the model output is traceable to real tool data".

#### 3.3.1 Fact Extraction

`FactExtractor` extracts 10 fact categories from both **tool call trace** and **model output**:

| Category | Tool-side Extraction | Output-side Extraction | Matching Method |
|----------|---------------------|----------------------|-----------------|
| flights | Regex `[A-Z]{2}\d{3,4}` from search_flights results | Same regex from output | Set intersection |
| trains | Regex `[GDCZTK]\d{1,5}` from search_train_tickets results | Same regex from output | Set intersection |
| pois | `名称:` pattern + `【】`/`「」`/JSON "name" | Same pattern from output | **Fuzzy match** (exact/normalized/half) |
| weather | Weather condition words (sunny/cloudy/rain/snow etc.) + temperature `N度` | Extract only from paragraphs with weather context (avoid false positives from POI names) | Set intersection |
| distances | `Nm`/`Nkm`/`N公里` (filter < 100m micro-segments) | Same regex | Text containment |
| times | `HH:MM` format + range format `HH:MM-HH:MM` | `HH:MM` in transport context | Set intersection |
| prices | `N元` + prices associated with transport IDs | Same regex + ID-associated prices | Set intersection (stringified comparison) |
| wind_info | `X风` + `N级` | Same regex | Set intersection |
| travel_durations | `(耗时\|用时)N(秒\|分钟\|小时)` | Same regex | Text containment |
| road_names | `X(路\|街\|大道\|高速\|环路)` (≥ 3 chars) | Same regex | Text containment |

#### 3.3.2 Per-Category Scoring

For each non-empty category:
```python
overlap_ratio = matched / min(len(tool_facts), max(1, len(output_facts)))
normalized    = min(1.0, overlap_ratio / 0.6)   # 60% overlap = full score
```

Where `matched` is computed differently per category:
- flights/trains/weather/times/prices/wind_info → **Set intersection** `|tool ∩ output|`
- pois → **Fuzzy match count** `sum(1 for poi in tool_pois if fuzzy_match(poi, output))`
- distances/travel_durations/road_names → **Text containment count** `sum(1 for d in tool_facts if d in output)`

#### 3.3.3 Minimum Quantity Threshold (Anti-Hack)

When tool-returned facts in a category ≥ `IC_MIN_QUANTITY_THRESHOLD`(5), the model must match at least `IC_MIN_QUANTITY_RATIO`(20%) of facts (capped at `IC_MIN_QUANTITY_CAP`=3). Failure caps that category's score at 65%.

```python
if len(tool_facts) >= 5:
    required = min(3, ceil(len(tool_facts) * 0.2))
    if matched_count < required:
        category_score *= 0.65  # IC_BELOW_MINIMUM_SCALE
```

#### 3.3.4 Context-Sensitive Matching

Flight/train facts use context-sensitive matching: facts not near relevant context keywords have their weight reduced to 50% (`IC_OUT_OF_CONTEXT_WEIGHT`=0.5). Prevents models from stacking facts in irrelevant positions.

#### 3.3.5 Aggregation & Breadth Penalty

```python
IC = 25 × (sum(normalized_scores) / num_categories_with_data)

# Breadth penalty: citing too few categories → ×0.5
if num_categories_with_data >= 4:  # INFO_CONSISTENCY_MIN_BREADTH_TOTAL
    min_breadth = max(2, (num_categories_with_data + 1) // 2)
    if categories_matched < min_breadth:
        IC *= 0.5  # IC_BREADTH_PENALTY_MULTIPLIER
```

**Edge cases**:
- Tools returned no data (`tool_facts.is_empty()`) → IC = 25 × 0.5 = 12.5 (half credit)
- No tool calls → IC = 0

### 3.4 Completeness (25 pts)

Measures "whether the output covers all necessary planning dimensions". Each problem type has different dimension allocations.

#### 3.4.1 Two Verification Functions

**`_check_with_grounded_context`** (standard dimensions, proximity-based anti-echo):

```
Input: text, keyword, context, tool_facts_set, max_pts, target_count

Proximity-based scoring (no free tier):
  keyword + context + tool_fact (proximate ≤500 chars)  → 100% × max_pts
  keyword + tool_fact (proximate ≤500 chars)            →  50% × max_pts
  keyword + tool_fact (distant >500 chars)              →  20% × max_pts (anti-echo)
  keyword + context (no tool_fact)                      →   0%          (no evidence)
  keyword only                                          →   0%          (no evidence)
  No tool data available                                →  10% × max_pts (structural credit)

Quantity scaling (target_count > 0):
  grounded_count = tool facts present in output and proximate
  tier_score *= grounded_count / target_count  (linear, no floor)
  → more tool facts cited = higher score

Budget/tips with no price data:
  → max 10% structural credit (STRUCTURAL_CREDIT_RATIO)
```

**`_check_with_verified_context`** (transport ID dimensions, stricter):

```
Input: text, keyword, verified_ids (flight/train IDs from tools), max_pts, target_count

Uses lookbehind+lookahead regex for exact ID matching:
  pattern = (?<![A-Za-z\d]) + ID + (?!\d)
  → prevents "AG102" matching "G102", or "G1023" matching "G102"

Scoring:
  keyword + at least 1 exact ID match → max_pts
  keyword + 0 matched IDs             → 0 (all fabricated = no credit)

Quantity scaling: matched_ids / target_count (min 25%)
```

#### 3.4.2 Dimension Allocations by Problem Type

**intercity (inter-city transport, 25 = 5+5+5+5+5)**

| Dimension | Points | Verification | keyword | Grounding Source | Target |
|-----------|--------|-------------|---------|-----------------|--------|
| Flight recommendations | 5 | verified_context | `(航班\|飞机\|机票)` | `tool_facts.flights` | 2 |
| Train recommendations | 5 | verified_context | `(火车\|高铁\|动车\|车次)` | `tool_facts.trains` | 2 |
| Departure/arrival times | 5 | grounded_context | `(出发\|到达\|发车\|起飞)` | `time_strs` | 3 |
| Price information | 5 | grounded_context | `(价格\|费用\|票价)` | `price_strs` | 3 |
| Recommendations | 5 | grounded_context | `(推荐\|建议\|最佳)` | `tool_facts.pois` (fuzzy) | 2 |

**multiday (multi-day tour, 25 = 5+5+4+4+4+3)**

| Dimension | Points | Verification | Description |
|-----------|--------|-------------|-------------|
| Day structure | 5 | Progressive POI grounding | Must match POIs for baseline score (no POI = 0) |
| Attraction arrangement | 5 | grounded_context | keyword `(景点\|游览\|参观)` + POI fuzzy match, target=days×2 |
| Dining recommendations | 4 | grounded_context | keyword `(餐\|吃\|美食)` + POI fuzzy match, target=days |
| Accommodation | 4 | grounded_context | keyword `(住宿\|酒店\|宾馆)` + POI fuzzy match, target=days-1 |
| Transportation | 4 | grounded_context | keyword `(交通\|出行)` + distances/durations |
| Budget breakdown | 3 | grounded_context | keyword `(预算\|费用\|花费)` + price_strs |

**hybrid (comprehensive, 25 = 6+5+4+4+3+3)**

| Dimension | Points | Verification | Grounding Source |
|-----------|--------|-------------|-----------------|
| Transport plan | 6 | verified_context | flights ∪ trains (exact IDs) |
| Day structure | 5 | Progressive POI grounding | Same as multiday |
| Attractions | 4 | grounded_context | POI names (fuzzy) |
| Dining | 4 | grounded_context | POI names (fuzzy) |
| Budget total | 3 | grounded_context | price_strs |
| Weather info | 3 | grounded_context | weather facts |

**single_poi (single attraction deep dive, 25 = 6+5+5+5+4)**

| Dimension | Points | Verification | Grounding Source |
|-----------|--------|-------------|-----------------|
| Tour arrangement | 6 | grounded_context | POI names (fuzzy) |
| Nearby recommendations | 5 | grounded_context | POI names (fuzzy) |
| Transport distance | 5 | grounded_context | distances ∪ durations |
| Tickets/tips | 5 | grounded_context | prices ∪ times (no data → POI+distance fallback) |
| Budget estimate | 4 | grounded_context | price_strs |

**food_tour (culinary tour, 25 = 6+5+5+5+4)**

| Dimension | Points | Verification | Grounding Source |
|-----------|--------|-------------|-----------------|
| Food/restaurants | 6 | grounded_context | POI names (fuzzy) |
| Dish recommendations | 5 | grounded_context | POI names (fuzzy) |
| Route order | 5 | grounded_context | distances ∪ durations |
| Cost estimate | 5 | grounded_context | price_strs (no data → POI+distance fallback) |
| Tips | 4 | grounded_context | POI ∪ weather |

**business (business travel, 25 = 6+5+4+5+5)**

| Dimension | Points | Verification | Grounding Source |
|-----------|--------|-------------|-----------------|
| Transport plan | 6 | verified_context | flights ∪ trains (exact IDs) |
| Hotel recommendations | 5 | grounded_context | POI names (fuzzy) |
| Dining recommendations | 4 | grounded_context | POI names (fuzzy) |
| Cost estimate | 5 | grounded_context | price_strs |
| Business facilities | 5 | grounded_context | POI names (fuzzy) |

**family_study (family/educational, 25 = 5+5+5+5+5)**

| Dimension | Points | Verification | Grounding Source |
|-----------|--------|-------------|-----------------|
| Day structure | 5 | Progressive POI grounding | Same as multiday |
| Family content | 5 | grounded_context | POI names (fuzzy) |
| Educational experiences | 5 | grounded_context | POI names (fuzzy) |
| Dining/accommodation | 5 | grounded_context | POI names (fuzzy) |
| Budget breakdown | 5 | grounded_context | price_strs (no data → POI+distance fallback) |

### 3.5 Fabrication Penalty (0 ~ -12.5)

Deducted from code score, composed of `ClaimVerifier` and transport fabrication detection. **Short outputs (< 200 chars) skip fabrication detection** since they are too brief for meaningful fabrication judgment.

#### 3.5.1 Price Fabrication Detection

For prices associated with transport IDs in the output (handled by `TransportGroundingVerifier`), and other non-transport prices:
- Found matching price in tool results → error > 10% → **-3.0 pts/instance**
- Transport ID-associated prices are handled by transport_grounded HC, skipped here

#### 3.5.2 Weather Fabrication Detection

- Weather condition words in output - weather condition words from tools = fabricated weather → **-2.0 pts**
- Only extracted from weather-context paragraphs (avoids false positives from POI names like "断桥残雪")

#### 3.5.3 Transport Fabrication Additional Deduction

```python
if transport_grounding enabled and total_transport_claims > 0:
    fab_ratio = unverified / total
    if fab_ratio > 0.1:
        additional_penalty = -5.0 × fab_ratio    # 10% fabrication → -0.5, 100% → -5.0
        penalty = max(penalty + additional_penalty, -12.5)
```

#### 3.5.4 Low IC Amplification

If `info_consistency / 25 < 0.4` (i.e., IC < 10 pts) and existing fabrication deduction > 3 pts → penalty capped at -10.0 (amplified punishment).

#### 3.5.5 POI Administrative Name Exclusion

POI extraction excludes administrative division names (province/city/district names) using regex negative lookahead `pname|cityname|adname`, preventing city names like "北京" from being misidentified as POIs.

### 3.6 LLM Semantic Evaluation (50 pts)

Uses an independent LLM to semantically evaluate the output. **UnifiedScorer** evaluates all 5 dimensions in a single API call. When LLM is unavailable, total score no longer zeros out — only code score is used (max 50 pts).

#### 3.6.1 Evaluation Models & Retry Strategy

```
Model list (LLM_MODELS):
  1. Qwen/Qwen3-235B-A22B-Instruct-2507-TEE  (retry 2×, interval 1s/2s)
  2. Qwen/Qwen2.5-72B-Instruct        (retry 1×)
  3. Qwen/Qwen3-32B                    (retry 1×)
         ↓ all failed
  Return: LLMEvaluationResult(success=False, error="All N models failed")
```

Each evaluation has **independent retries**, no global circuit breaker (removed). No API Key → returns error directly.

#### 3.6.2 Structured Summary Architecture

**The LLM evaluator never sees the raw model output**. Evaluation code first extracts structured data, then submits it to the LLM:

```
Model output → FactExtractor → Structured summary (structured_summary)
                             → Tool facts summary (facts_summary)
                                    ↓
                         UnifiedScorer LLM evaluation
                         (scores based on structured data only)
```

The structured summary contains:
- **Basic statistics**: output length, days planned
- **Transport info**: flights/trains + tool-verified(✓/✗) + price match
- **Attractions/POIs**: with/without tool evidence + match rate
- **Price info**: each price + verification status (✓/✓subtotal/?)
- **Weather info**: temperature/weather match (✓/✗)
- **Reasoning quality**:
  - Number and content of reasoning fragments
  - Comparison claims vs tool-comparable options (hollow comparison detection)
  - Data limitation awareness (✓/✗)

#### 3.6.3 Five Evaluation Dimensions

Each dimension scored 0-10 by LLM, converted by weight: `score = raw × weight / 10.0` (weight=10 → max 10)

| Dimension | Max | Evaluation Criteria |
|-----------|-----|-------------------|
| **practicality** | 10 | Reasonable time scheduling, smooth transport connections, no obvious conflicts |
| **analysis_depth** | 10 | Distinguishes data recitation from real analysis; penalizes echo/hollow comparisons; rewards limitation awareness (+2) and data computation (+1) |
| **logic** | 10 | Reasonable geographic grouping of routes; penalizes planning based on fabricated POIs; penalizes unsourced distances/times |
| **user_experience** | 10 | Explicitly responds to all user constraints and preferences, reasonable budget allocation, trade-off explanation for conflicting constraints |
| **factual_grounding** | 10 | Are flights/trains/prices/POIs traceable to tool data? Short outputs score-limited |

#### 3.6.4 Anti-Cheat Measures

| Measure | Description |
|---------|-------------|
| **Structured summary** | LLM never sees raw output, only code-extracted structured data — eliminates prompt injection |
| **Rubric randomization** | Each penalty value randomized within config range (e.g., no_transport_mode: 2-4), prevents memorizing scoring criteria |
| **Dimension order randomization** | 5 dimensions shuffled randomly in the prompt each time |
| **Boundary token** | Random UUID wraps output content, prevents boundary injection |
| **Output sanitize** | 26 regex patterns filter injection attempts (ignore above, set all scores, etc.) |
| **Cross-validation** | total > 36/50 → second model re-evaluates, take per-dimension min (conservative) |
| **Hollow comparison detection** | Comparison claims far exceeding tool-comparable options → flagged warning |
| **Echo detection guidance** | analysis_depth rubric explicitly distinguishes "data copying" from "real analysis" |
| **Limitation awareness** | Proactively noting tool data limitations → +2 bonus (key differentiator: real analysis vs echo) |
| **Fabricated POI planning penalty** | Geographic grouping based on non-tool POIs → logic deduction |
| **Price subtotal recognition** | 850+150=1000 marked as "✓subtotal", avoids false fabrication flag |

#### 3.6.5 Bidirectional Coupling

```python
# 1. LLM constrained by code (low code → LLM compressed)
code_ratio = min(1.0, code_total / (50 × 0.6))   # code_total / 30
llm_adjusted = llm_raw × code_ratio
#   code = 0     → llm_adjusted = 0      (LLM score fully nullified)
#   code = 15    → llm_adjusted = 50%    (linear compression)
#   code = 30+   → llm_adjusted = 100%   (full credit)

# 2. Code constrained by LLM (low LLM → code discounted, min 70%)
if llm_validation_success:
    llm_ratio = min(1.0, llm_total / (50 × 0.4))   # llm_total / 20
    code_coupled = code_total × (0.7 + 0.3 × llm_ratio)
#   llm = 0     → code × 0.70   (code discounted to 70%)
#   llm = 20    → code × 1.00   (full credit)

# 3. Monotonicity guarantee (coupling never drops below 50% of raw sum)
base = max(code_coupled + llm_adjusted, 0.5 × raw_sum)
```

- Direction 1 ensures models can't score high LLM via "eloquent but unsupported by tools"
- Direction 2 ensures models can't score high code via "data dump but unreadable"
- Monotonicity guarantee prevents extreme coupling from causing score cliffs

#### 3.6.6 Defense-in-Depth (LLM-Code Double Insurance)

When LLM detects severe fabrication (factual_grounding < 4/10) but code didn't penalize enough, additionally compress IC and Comp:

```python
if factual_grounding < 4.0:
    llm_grounding_mult = 0.3 + 0.7 × (factual_grounding / 4.0)
    combined = min(local_fab_multiplier, llm_grounding_mult)
    # Compress info_consistency and completeness
```

### 3.7 Complete Scoring Path Summary

| Path | Trigger | Code | LLM | Total | Description |
|------|---------|------|-----|-------|-------------|
| P1 | `tool_info_used=False` (transport) | Computed | Skipped | 0 | IC below threshold, hard fail |
| P1b | `tool_info_used=False` (non-transport) | Computed | Optional | ~base×0.05 | IC below threshold, softened |
| P2 | `format_valid=False` | Computed | Optional | ~base×0.15 | Preserves RL gradient |
| P3 | LLM available + normal flow | Computed | Filled | 0-100 | Full scoring (code + LLM) |
| P4 | LLM unavailable + normal flow | Computed | 0 | 0-50 | Code score only, error logged |
| P5 | No API Key | Computed | 0 | 0-50 | Same as P4, no zeroing |

### 3.8 Anti-Cheat System

| Mechanism | Defense Target | Implementation |
|-----------|---------------|----------------|
| Proximity-based grounding | Keyword echo/data copying | Tool facts must be within ≤500 chars of keyword for full score; distant = 20% only (anti-echo) |
| Exact ID matching | Fabricated flights/trains | lookbehind+lookahead regex `(?<![A-Za-z\d])G1234(?!\d)` |
| IC min quantity threshold | Minimal citation gaming | Tool returns ≥4 facts → must match ≥30% (max 3), else capped at 50% |
| IC context-sensitive | Fact stacking | Facts not near relevant context → weight reduced to 50% |
| Breadth penalty | Single-category citing | Matched categories < half of total and available categories ≥ 3 → IC × 0.3 |
| Fabrication deduction | Hallucination | Fabricated transport/prices/weather → max -12.5 pts |
| Progressive transport penalty | Partial fabrication | Fabrication ratio 20%→100%, multiplier 1.0x→0.3x (linear interpolation) |
| Epoch Salt + cache alignment | Memorizing historical data | Weekly TRANSPORT_SALT rotation; AMap TTL aligned to weekly epoch |
| LLM-Code Coupling | Optimizing only LLM score | code < 30 → LLM linearly compressed |
| Defense-in-depth | Code+LLM double insurance | fg<4/10 → compress IC/Comp, take min with code-side fabrication detection |
| Pure code tool_info_used | Hacking score without tools | IC≥6 (transport) / IC≥8 (non-transport), IC-only, no Comp |
| Structured summary | Prompt injection | LLM never sees raw output, only code-extracted structured data |
| Echo detection | Data copying disguised as analysis | analysis_depth rubric distinguishes recitation vs analysis; hollow comparison detection |
| Fabricated POI planning penalty | Using external knowledge to fake plans | logic rubric penalizes geographic planning based on non-tool POIs |
| Rubric randomization | Memorizing scoring criteria | Penalty values/dimension order/calibration anchors randomized each time |
| Cross-validation | Single model bias | total>36/50 → second model re-evaluates, take min |
| Structural credit limit | Fabricating without data | Max 10% structural credit when no tool data |
| Day grounding | Fabricated itinerary | Day structure needs POI matches for baseline, no POI = 0 |
| POI admin exclusion | City name as POI false positive | POI extraction excludes administrative division names |
| Anti-injection filter | Prompt Injection | 26 regex + random boundary token + explicit warnings |
| Quantity scaling | Minimal citation gaming | `grounded_count / target_count` linear scaling (no floor) |

### 3.9 Return Value Structure

#### 3.9.1 Default Mode (`to_safe_dict`, external/RL consumers)

To prevent models from reverse-engineering scoring details, the default returns obfuscated scores:

```json
{
  "total": 42.5,
  "code_band": "medium",
  "llm_band": "high",
  "hard_constraints": {"format_valid": true, "tool_info_used": true, ...},
  "noisy_code": 23.1,
  "noisy_llm": 19.4,
  "llm_available": true
}
```

- `code_band`/`llm_band`: 5-tier bands (very_low/low/medium/high/very_high)
- `noisy_code`/`noisy_llm`: Noised aggregate scores (σ=2.0 Gaussian noise)
- Per-dimension exact scores not exposed

#### 3.9.2 Debug Mode (`QQR_DEBUG=1` → `to_dict`)

```json
{
  "total": 42.5,
  "code_score": {
    "info_consistency": 18.5,
    "completeness": 20.0,
    "fabrication_penalty": -2.0,
    "subtotal": 36.5,
    "ic_categories": {...},
    "comp_subscores": {...}
  },
  "llm_score": {
    "practicality": 6.0,
    "analysis_depth": 8.0,
    "logic": 7.0,
    "user_experience": 5.0,
    "factual_grounding": 9.0,
    "subtotal": 35.0,
    "coupled_subtotal": 33.2,
    "reasons": {...}
  },
  "hard_constraints": {...},
  "parse_success": true,
  "llm_validation_success": true
}
```

---

## 4. Problem Generation System

### 4.1 Seven Problem Types

```
task_id % 7 → problem type:
  0: intercity      Inter-city transport   (requires: poi_search + direction + weather + transport*)
  1: multiday       Multi-day tour         (requires: poi_search + around_search + direction + weather)
  2: hybrid         Comprehensive          (requires: all 6 tools)
  3: single_poi     Single attraction      (requires: poi_search + around_search + direction + weather)
  4: food_tour      Culinary tour          (requires: poi_search + around_search + direction + weather)
  5: business       Business travel        (requires: poi_search + direction + weather + flights + trains)
  6: family_study   Family/educational     (requires: poi_search + around_search + direction + weather)

* intercity transport tools determined by distance category:
  short  → search_train_tickets (train only, some cities lack airports)
  medium → search_flights + search_train_tickets (both available)
  long   → search_flights (flights only)
```

### 4.2 Deterministic Generation

```python
rng = random.Random(task_id)  # task_id as seed

# All parameters deterministically derived from rng:
problem_type = PROBLEM_TYPES[task_id % 7]
difficulty = (task_id // 7) % 3 + 1        # 1/2/3 cycling
destination = rng.choice(MAJOR_CITIES)      # from 70+ cities
travel_date = base_date + timedelta(days=task_id % 365)
interests = rng.sample(INTERESTS, rng.randint(2, 4))
# ...
```

**Dynamic transport description**: `_intercity_to_prompt()` dynamically selects transport mode description in prompts based on `required_tools`. Short-haul includes only `search_train_tickets`, so the prompt mentions only "train"; long-haul includes only `search_flights`, mentioning only "flights"; medium includes both. This prevents models from calling unavailable transport tools (e.g., flight search for cities without airports).

### 4.3 Difficulty Levels & DifficultyProfile

| Level | Label | Tools | Max Days | DifficultyProfile |
|-------|-------|-------|----------|-------------------|
| 1 | beginner | 2-3 | 1 | constraint_tightness=0.5, conflicts=1, time_pressure=False |
| 2 | intermediate | 3-5 | 3 | constraint_tightness=0.75, conflicts=2, time_pressure=possible |
| 3 | advanced | 5-6 | 5 | constraint_tightness=0.95, conflicts=3, time_pressure=True |

**DifficultyProfile** controls:
- `constraint_tightness` — Budget tightening factor (higher = tighter)
- `constraint_conflicts` — Number of injected conflicting constraint pairs (e.g., "budget priority" + "comfort priority")
- `time_pressure` — Whether there's an urgent arrival time window

**Budget floor protection**: `_apply_budget_tightness()` ensures after tightening, budget doesn't fall below `MIN_BUDGET_PER_PERSON_DAY × days × travelers + MIN_TRANSPORT_COST[distance_type] × travelers`. `MIN_TRANSPORT_COST` sets minimum transport cost per distance category (short=50, medium=150, long=300 CNY/person/one-way), conservatively estimated from mock_transport pricing, preventing mathematically unsolvable problems.

### 4.4 City Pair Safety Validation

`config.py` maintains `CITIES_WITHOUT_AIRPORTS` and `CITIES_WITHOUT_TRAINS` sets, recording cities known to lack airports or train stations. Module-level assertions at import time verify every city in `CITY_PAIRS` has at least one transport mode (airport or train station), preventing maintainers from adding cities with no transport options.

### 4.5 City Knowledge Graph

`knowledge_graph.py` provides structured information for all 71 cities:

```python
CityProfile:
  specialties   # City characteristics (e.g., "historical culture", "natural scenery")
  landmarks     # 2-6 famous attractions
  food_themes   # Matching food theme keywords
  seasonal_avoid # Months to avoid (typhoon season, extreme heat/cold)
  transport_hub  # Whether it has major airport + high-speed rail station
  nearby_cities  # Nearby cities suitable for multi-day extensions
```

Used for:
- Seasonal checks: Avoid generating travel problems for inappropriate seasons
- Interest biasing: Bias interest keywords toward city characteristics
- Food theme matching: food_tour type selects cities matching food themes
- POI pool: Landmark data used for single_poi type attraction selection

---

## 5. Deterministic Transport Data Generation

`mock_transport/server.py` generates flight and train data based on SHA256 seeds:

```python
# Seed construction
seed = SHA256(f"{TRANSPORT_SALT}|{date}|{from_city}|{to_city}")

# Deterministically derived from seed:
- Flight count (8-15), airlines (15), flight numbers (e.g., CA1234), prices, departure/arrival times
- Train count (8-15), train types (G/D/C/Z/T/K), train numbers (e.g., G1986), prices, times
- Flight distance (city pair lookup or SHA256 fallback, symmetric and salt-independent)
```

**Key design choices**:
- `TRANSPORT_SALT` auto-rotates weekly: `str(int(time.time()) // (7 * 86400))`
- AMap cache TTL synced with TRANSPORT_SALT to weekly epoch boundary, ensuring all data sources stable within same epoch
- Flight number deduplication: no duplicate IDs within a query
- Distance symmetry: `distance(A→B) = distance(B→A)`
- Fallback distance is salt-independent: `SHA256("distance|sorted_city_pair")`
- 70+ city airport/station mappings: uses real names (e.g., "Capital International Airport", "Beijing South Station")
- 6 train types filtered by distance: short-haul only G/D/C, long-haul includes Z/T/K
- Red-eye flights: 1-2 red-eye flights per generation, with price discounts

---

## 6. Output Parser

`parser.py` parses model output into structured data (`ParsedOutput`):

1. **JSON-first**: Attempts to extract JSON from code blocks or raw text
2. **Regex Fallback**: If no JSON, extracts via regex:
   - Transport options: `Flight CA1234, price XXX yuan` pattern
   - Daily itinerary: `Day N` segmentation
   - Budget: Category keywords + price patterns
   - Locations: Suffix matching (scenic area, park, museum, ancient town, etc.)

---

## 7. Quick Start

### 7.1 Environment Requirements

```bash
# Required in .env file:
CHUTES_API_KEY=cpk_...       # Chutes LLM API (for model under test and LLM evaluator)
AMAP_MAPS_API_KEY=a605...    # AMap API (for POI/navigation/weather)
```

### 7.2 Build & Run

```bash
# Activate virtual environment
source .venv/bin/activate

# Build Docker image
afs build environments/qqr --tag qqr:v1

# Start container
afs run qqr:v1 --name qqr --env CHUTES_API_KEY=$CHUTES_API_KEY --env AMAP_MAPS_API_KEY=$AMAP_MAPS_API_KEY

# Single evaluation
afs call qqr evaluate --arg task_id=131

# Batch evaluation
python examples/qqr/test_qqr.py
```

### 7.3 Python API

```python
import affinetes as af

env = af.load_env(
    image="qqr:v1",
    mode="docker",
    env_vars={
        "CHUTES_API_KEY": api_key,
        "AMAP_MAPS_API_KEY": amap_key,
    },
)

# Full evaluation (internally handles two-phase Agent Loop)
result = await env.evaluate(
    model="moonshotai/Kimi-K2.5-TEE",
    base_url="https://llm.chutes.ai/v1",
    task_id=131,
    timeout=300,
    temperature=0.7,
)

print(f"Score: {result['score'] * 100:.1f}/100")
print(f"Pass: {result['success']}")  # score >= 60 is pass

# Manual Agent Loop control (suitable for RL training)
reset_resp = await env.reset(task_id=131)
episode_id = reset_resp.episode_id

# Phase 1: Tool calling (standard OpenAI tool_calls format)
step_resp = await env.step(
    action="",
    episode_id=episode_id,
    tool_calls=[{
        "id": "call_abc123",
        "type": "function",
        "function": {
            "name": "poi_search",
            "arguments": '{"address": "贵阳", "region": "贵阳"}'
        }
    }],
)
print(f"Step reward: {step_resp.reward}")  # 0~1

# Phase 2: Final answer (no tool_calls → triggers scoring)
final_resp = await env.step(
    action="Complete travel plan...",
    episode_id=episode_id,
    tool_calls=None,
)
print(f"Final score: {final_resp.info['score']}")  # 0~100
```

### 7.4 Actor Initialization Parameters

```python
Actor(
    enable_llm_validator=True,          # Whether to enable LLM semantic evaluation
    llm_validator_model="Qwen/Qwen3-235B-A22B-Instruct-2507-TEE",  # Preferred LLM evaluator model (falls back to LLM_MODELS list)
)
```

### 7.5 Dependencies

```
httpx>=0.25.0
openai>=1.0.0
openai-agents>=0.6.0
mcp>=1.0.0
diskcache>=5.6.0
click>=8.0.0
```
