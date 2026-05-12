# Project Brief: Customer Support Query Routing

**Client:** Meridian Bank (mid-sized retail bank)
**Prepared by:** Customer Operations & Digital Channels
**Date:** May 2026
**Status:** Discovery / scoping

---

## 1. Background

Meridian Bank serves approximately 2 million retail customers across personal banking, cards, and lending products. Our customer support function handles inbound queries through three channels: live chat (web and mobile app), email, and phone. Over the last 18 months, query volume in the chat channel has grown roughly 40% year-over-year, driven mainly by mobile app adoption.

Our current routing process for chat queries works as follows: an incoming message lands in a shared queue, a tier-1 agent picks it up, reads it, and either resolves it directly (about 55% of cases) or escalates it to a specialist queue (cards fraud, lending, account management, technical support, etc.). The tier-1 agent is effectively performing manual triage on every incoming message before any real work happens.

Average time from message arrival to agent first response is currently 4 minutes 20 seconds, against an internal SLA of 90 seconds. Customer satisfaction scores have declined two consecutive quarters, and exit survey data indicates wait time is the top driver.

## 2. The opportunity we're exploring

We believe a meaningful portion of the tier-1 triage work could be automated. If incoming messages could be classified by intent before reaching an agent, we could:

- Route specialist queries directly to the right team, skipping the tier-1 hop entirely
- Surface a suggested response template to the agent for high-confidence routine queries
- Get better data on what customers are actually asking about (currently we have only the agent's after-the-fact category tag, which is inconsistent)

We are explicitly **not** trying to replace agents or automate end-to-end conversations at this stage. The goal is to make the existing agents faster and better-targeted.

## 3. What we've tried so far

Our innovation team ran a six-week proof of concept earlier this year. They built a prototype that calls a frontier large language model (we used a major commercial provider's flagship model) to classify each incoming message into one of ~80 internal intent categories.

The results were encouraging:
- Accuracy on a hand-labeled sample of 500 messages was around 91%
- Agents who saw the predicted intent in their UI reported it was usually right and useful

However, the prototype is not viable for production for several reasons:

- **Cost:** At our current chat volume (~30,000 messages/day and growing), the per-query cost of the frontier model puts the annual bill in a range that finance will not approve for a triage feature
- **Latency:** End-to-end response times from the LLM provider averaged 1.8 seconds, occasionally spiking to 5+ seconds. Agents found the delay disruptive
- **Data handling:** Sending customer messages to an external LLM provider raises questions from our Compliance and Information Security teams that we have not fully resolved. Several categories of message contain account numbers, transaction details, or other sensitive information
- **Operational dependency:** A production feature in our agent workflow depending on a third-party API is a new pattern for us and our Ops team has concerns about uptime, rate limits, and vendor lock-in

We need a path forward that preserves the accuracy we saw in the proof of concept but addresses these issues.

## 4. What we'd like from this engagement

We're engaging Datatonic to help us design and prototype a production-ready approach. We're open on the technical solution. Specifically, we'd like:

- A proposed approach with a clear rationale for the choices
- A working prototype on a representative dataset that demonstrates the approach is viable
- Evidence (numbers, not just claims) that the approach meets our constraints
- A clear-eyed view of what works, what doesn't, and what would need to happen to take this from prototype to production
- A recommended architecture on Google Cloud (we are a GCP shop)

## 5. Constraints and requirements

### Functional

- The system must classify incoming chat messages into one of our internal intent categories (we'll provide the taxonomy; expect ~70-80 categories)
- The system must return a confidence signal that the downstream routing logic can act on (e.g., high-confidence → auto-route, low-confidence → human triage)
- The system must expose a simple API that our existing chat platform can call

### Non-functional

- **Cost:** Target per-query cost at least 10x lower than the frontier-LLM prototype. Ideally closer to 50x lower
- **Latency:** P95 end-to-end latency under 300ms, measured at the API boundary
- **Accuracy:** Should not regress meaningfully from the frontier-LLM baseline. We're willing to accept a small drop in exchange for cost and latency wins, but the size of that acceptable drop is something we'd want to discuss based on what's achievable
- **Throughput:** Must handle current peak volume (~80 messages/second during morning rush) with headroom for 3x growth

### Compliance and data handling

- All processing must occur within our GCP organization. No customer message content can leave our cloud perimeter
- We can supply historical chat transcripts for development purposes under our internal data governance process, but they must be handled appropriately
- The system's decisions must be auditable. For any classified message, we need to be able to retrieve what the system predicted and why

### Operational

- Whatever is built must be reproducible — we should be able to retrain, redeploy, and roll back without bespoke manual steps
- Our MLOps maturity is moderate. We have a small platform team but not a dedicated ML platform org. The solution should be operable by a team of 2-3 people, not require a standing team of 10
- We have a strong preference for Google Cloud managed services over self-hosted alternatives where the trade-offs make sense

## 6. Out of scope (for now)

To keep this engagement focused, the following are explicitly out of scope:

- Full conversational automation or chatbots that resolve queries end-to-end
- Voice channel (phone calls) — chat only for this phase
- Languages other than English — our current chat volume is ~95% English; we'll address other languages in a future phase
- Sentiment analysis, customer profiling, or any cross-message inference beyond per-message intent
- Changes to the agent UI or to the underlying chat platform itself

## 7. Success criteria

We will consider this engagement successful if, at the end of it, we have:

- A documented technical approach that the Datatonic team would recommend taking to production
- A working prototype demonstrating the approach on representative data
- Measured results on the cost, latency, and accuracy constraints above, with honest discussion of where the approach falls short
- A clear plan for what production deployment would entail beyond the prototype, including any gaps or risks

## 8. Open questions we'd expect to discuss

We don't have answers to all of these yet and would expect the Datatonic team to help us think through them:

- How should we handle the "long tail" of intent categories that have very few examples in our historical data?
- What's the right fallback when the system isn't confident? Send to tier-1 as today, or some other handling?
- How do we plan to handle drift — new product launches, new query patterns, etc.?
- What does the retraining cadence look like, and what triggers it?
- How do we validate the system before rolling it out to real customer-facing traffic?

---

*This brief is a starting point. We expect the scope and details to evolve through discovery.*