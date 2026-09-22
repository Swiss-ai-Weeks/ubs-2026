# Transaction Activity Forecasting

> How can we use AI to predict a client's future transaction activity from
> their historical financial transaction patterns?

Clients generate rich streams of everyday banking activity: salary deposits,
subscriptions, utility and other household bills, loan repayments, card
purchases, transfers, and cash withdrawals. Some of these transactions are
one-off, while others follow recognizable patterns and form recurring payment
streams that reflect a client's financial commitments and lifestyle.

Understanding and anticipating these recurring transactions can help clients
stay in control of their finances: planning ahead for regular outgoings,
avoiding missed payments, and spotting changes early. It also enables more
tailored support, such as personalized insights, relevant alerts, and guidance
aligned with each client's situation and goals, as well as earlier detection
of unusual activity that may indicate fraud or misuse.

In this challenge, you will work with a synthetic transaction dataset designed
to resemble real-world client behavior. Each record represents an individual
transaction and includes information such as timing, transaction type,
amount, merchant category, and description. Using historical transaction
data, your task is to predict the next recurring transaction a client is
expected to make.

A useful solution does not need to identify the exact merchant: predicting
the next likely recurring **merchant family** (e.g. mobile, insurance,
streaming) is enough to power reminders, cashflow forecasts, subscription
reviews, and proactive budgeting experiences.

The challenge requires participants to uncover recurring payment patterns
hidden within noisy transaction histories. Successful solutions will
recognize behavioral regularities, capture temporal relationships, and
generalize across diverse client spending and payment habits, ultimately
helping us anticipate client needs and support them more effectively.

## Objectives

- Build an AI pipeline that uses historical transaction data to predict a
  client's next recurring merchant family (e.g. cloud, gym, software).
- Identify and model recurring transaction patterns from raw transactional
  data, including temporal, behavioral, and merchant-related signals.
- Explore different approaches to feature engineering, sequence
  representation, and recurrence detection.
- Compare multiple modeling techniques and evaluate their performance using
  appropriate forecasting metrics.
- Analyze the strengths, limitations, and interpretability of their
  solution, and communicate key insights derived from the data.

## Why Hack?

This challenge lets you apply modern AI techniques to a realistic financial
time-series problem. You will explore how to turn raw transaction histories
into meaningful signals, design and compare different modeling approaches,
and think about what "good" prediction performance means in real-world
banking scenarios.

This is an opportunity to tackle a real-world financial problem and
demonstrate how AI can turn transactional data into actionable insights: the
same merchant-family predictions you build here are what could power
reminders, cashflow forecasts, subscription reviews, and proactive budgeting
experiences for real clients.

## Task Specification

Build a supervised prediction workflow that predicts the **next recurring
merchant family** for each client after the cutoff date (`2026-01-01`).

- Target column (`train_labels.csv` / `valid_labels.csv`):
  `target_next_recurring_merchant`
- Prediction column (your submission CSV):
  `predicted_next_recurring_merchant`

For each client, the model receives transaction history up to the cutoff date
and must forecast which merchant family (if any) recurs within the 90-day
horizon after that cutoff, maximizing `macro-F1` (arithmetic mean of class-wise F1 scores) across the label set below.

Allowed labels:

```text
cloud, gym, insurance, mobile, music, software, streaming, none
```

Use `none` when no recurring merchant family is expected to recur within the
90-day horizon.

## Data Package

You can find the following files in the `data` directory of this repository:

- `unlabeled_pretrain_transactions.jsonl`
- `train_transactions.jsonl`
- `train_labels.csv`
- `valid_transactions.jsonl`
- `valid_labels.csv`
- `test_transactions.jsonl`
- `sample_submission.csv`

Use the files as follows:

- `unlabeled_pretrain_transactions.jsonl`: optional extra transaction histories
  without labels. Use this only if you want to learn general transaction
  patterns before training a supervised model.
- `train_transactions.jsonl`: transaction histories for the training clients
  up to the cutoff date (labels are in `train_labels.csv`).
- `train_labels.csv`: target labels for the training clients.
- `valid_transactions.jsonl`: transaction histories for the validation
  clients up to the cutoff date (labels are in `valid_labels.csv`).
- `valid_labels.csv`: target labels for local validation and model selection.
- `test_transactions.jsonl`: hidden-test client histories up to the cutoff
  date. Use this to generate your milestone submission.
- `sample_submission.csv`: the exact test `client_id` set and required
  submission schema. Your submitted CSV must contain these clients.

### Data Shape

Transaction files are JSON Lines. Each line is one transaction event. Label and
submission files are CSV.

#### Transaction Fields

| Field | Type | Description | Example |
| --- | --- | --- | --- |
| `client_id` | string | Unique client identifier. | `C000001` |
| `timestamp` | string | Date and time the transaction occurred. | `2024-11-08T01:14:02Z` |
| `amount` | float | Transaction amount, in `currency`. | `275.94` |
| `currency` | string | Currency code for `amount`. | `eur` |
| `direction` | string | Whether funds moved into (`in`) or out of (`out`) the client's account. | `out` |
| `type` | string | Transaction type. | `card_payment` |
| `mcc` | string | Merchant category code. | `5411` |
| `description` | string | Free-text merchant/transaction description. | `neighborhood market` |
| `fee` | float | Fee charged for the transaction, in `currency`. | `0.0` |

`train_transactions.jsonl`

```jsonl
{"amount": 275.94, "client_id": "C000001", "currency": "eur", "description": "p2p receive", "direction": "in", "fee": 0.0, "mcc": "6012", "timestamp": "2024-11-08T01:14:02Z", "type": "p2p_transfer"}
{"amount": 143.27, "client_id": "C000001", "currency": "eur", "description": "neighborhood market", "direction": "out", "fee": 0.0, "mcc": "5411", "timestamp": "2024-11-11T06:25:52Z", "type": "card_payment"}
{"amount": 15.81, "client_id": "C000001", "currency": "eur", "description": "coffee shop", "direction": "out", "fee": 0.0, "mcc": "5812", "timestamp": "2024-11-12T15:45:16Z", "type": "card_payment"}
{"amount": 491.24, "client_id": "C000001", "currency": "eur", "description": "electronics shop", "direction": "out", "fee": 0.0, "mcc": "5732", "timestamp": "2024-11-14T09:39:25Z", "type": "card_payment"}
```

`train_labels.csv`

```csv
client_id,cutoff_date,target_next_recurring_merchant
C000001,2026-01-01,streaming
C000005,2026-01-01,mobile
C000007,2026-01-01,cloud
```

## Submissions

### Milestones

All deadlines are in Central European Summer Time (CEST), Zurich.

| Milestone | Deadline |
| --- | --- |
| 1 | Day 1 before 12:00 |
| 2 | Day 1 before 17:00 |
| 3 | Day 2 before 12:00 |
| 4 | Day 2 before 17:30 (submission to main jury) |

### Submission Contract

Create a submission file with the following schema:

```csv
client_id,predicted_next_recurring_merchant
C000004,none
C000008,none
```

- Use exactly the client IDs from `sample_submission.csv`, one row each.
- Use only the allowed labels listed in [Task Specification](#task-specification).
- Keep the required column names unchanged.

A submission is **valid** if it satisfies all of the rules above.

### Submission Workflow

Submit your predictions via [this form](https://forms.gle/3mgyM9D8d2quqXQ59):

- Your team name
- A link to your code repository
- Your submission file (matching the Submission Contract above)

You can submit multiple times before a milestone deadline; only your last
valid submission before the deadline is considered for scoring. A submission
made after a milestone deadline is instead taken into account for the next
milestone. Submissions cannot be made after the milestone 4 deadline.

### Scoring

Each team's final rank is based on the best macro-F1 achieved by any of
their valid milestone submissions across the whole event.
