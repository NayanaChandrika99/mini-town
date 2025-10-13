# Importance Scoring Rubric Guide

**Purpose**: Define clear, consistent criteria for rating observation importance (1-10) to ensure high-quality training data for DSPy compilation.

**Last Updated**: 2025-10-11
**Version**: 1.0

---

## Scoring Scale (1-10)

### Score 1-2: Trivial / Background Noise
**Definition**: Completely irrelevant to agent's goals. Environmental details with no actionable value.

**Examples**:
- "The grass is green today"
- "A clock is ticking in the background"
- "The sky is blue"
- "Someone's phone buzzed far away"

**Key Characteristics**:
- No connection to agent's personality or goals
- Cannot influence any decision-making
- Pure environmental description
- No emotional or social component

---

### Score 3-4: Mildly Interesting
**Definition**: Somewhat relevant but not immediately actionable. Provides context but doesn't require response.

**Examples**:
- "The weather is nice today" (for agent not focused on outdoor activities)
- "Someone walked past on the street"
- "Bob mentioned he likes coffee" (for agent not building relationships)
- "The store has a sale this weekend"

**Key Characteristics**:
- Peripheral relevance to goals
- No time-sensitivity
- Interesting information but not urgent
- Could be useful in future context

---

### Score 5-6: Relevant & Worth Remembering
**Definition**: Directly relates to agent's goals. Worth storing for future reference, but not immediately urgent.

**Examples**:
- "Bob mentioned he's working on a research project" (for agent with research goal)
- "Alice usually jogs in the morning" (for social agent building relationships)
- "The community garden needs volunteers next month"
- "Carol knows a lot about local history"

**Key Characteristics**:
- Clear connection to agent's stated goal
- Provides useful context for future interactions
- Not time-critical
- Could influence medium-term planning

---

### Score 7-8: Directly Impactful
**Definition**: Significantly impacts current plans or immediate goals. Requires attention and likely action.

**Examples**:
- "Alice invited me to a party at 7pm tonight" (for social agent)
- "The library closes in 30 minutes" (for agent needing to complete research)
- "Bob asked if I can help him tomorrow morning"
- "Carol mentioned a book I've been looking for"

**Key Characteristics**:
- Time-sensitive (today/this week)
- Directly affects current plan
- Requires decision or action
- Aligns strongly with personality and goals

---

### Score 9-10: Critical / Life-Changing
**Definition**: Urgent, high-priority situations requiring immediate attention. Safety-critical or major life events.

**Examples**:
- "Fire alarm going off in the building!"
- "Alice just got engaged"
- "Someone is calling for help"
- "You've been offered your dream job"
- "Emergency: your research deadline moved to tomorrow"

**Key Characteristics**:
- Immediate action required
- Safety or wellbeing implications
- Major life milestone
- Overrides all other priorities

---

## Context-Dependent Scoring

**CRITICAL**: The same observation can have different scores based on agent personality and goal!

### Example 1: "Alice invited me to a party tonight"

| Agent Profile | Score | Rationale |
|--------------|-------|-----------|
| Social, goal: "Build relationships" | 8 | Directly aligns with goal, time-sensitive, social event |
| Introverted, goal: "Complete research" | 4 | Interesting but conflicts with research priorities |
| Organized, goal: "Maintain garden" | 5 | Relevant for socializing but not primary goal |

### Example 2: "Found a book about local history"

| Agent Profile | Score | Rationale |
|--------------|-------|-----------|
| Analytical, goal: "Research local history" | 8 | Directly enables primary goal |
| Social, goal: "Build relationships" | 3 | Interesting but not relationship-focused |
| Detail-oriented, goal: "Document stories" | 7 | Very relevant to documentation goal |

---

## Edge Cases & Guidelines

### Ambiguous Observations
When observation lacks context, score conservatively:
- "Bob seems upset" → 6 (relevant for social agents, but uncertain)
- "Something is happening at the town square" → 5 (worth investigating but vague)

### Multiple Interpretations
Document reasoning for both interpretations:
- "The garden needs water" could be 4 (general observation) or 8 (for agent maintaining garden)

### Personality Modifiers
- **Impulsive agents**: Slightly higher scores for spontaneous opportunities (+1)
- **Risk-averse agents**: Higher scores for warnings/safety (-1 for risks)
- **Detail-oriented agents**: Higher scores for specific information (+1)

---

## Inter-Rater Agreement Guidelines

To ensure consistency when multiple people rate:

1. **Read full context**: Always consider agent_goal and agent_personality
2. **Use examples**: Reference similar rated examples
3. **Document reasoning**: Brief rationale for each score
4. **Allow ±1 variance**: Scores within 1 point are considered agreement
5. **Flag disagreements**: Any variance >2 points needs discussion

**Target**: Cohen's kappa > 0.6 (substantial agreement)

---

## Common Mistakes to Avoid

❌ **Scoring without considering agent context**
- Wrong: "Party invitation" always = 8
- Right: "Party invitation" = 8 for social agent, 4 for introverted researcher

❌ **Confusing interesting with important**
- Wrong: "Bob won the lottery" = 10 (interesting but not about MY agent)
- Right: "Bob won the lottery" = 3-4 (interesting gossip but not my life event)

❌ **Ignoring time-sensitivity**
- Wrong: "Meeting tomorrow" = 5 (sounds medium priority)
- Right: "Meeting tomorrow" = 7-8 (time-sensitive, requires planning)

❌ **Over-scoring mundane details**
- Wrong: "Nice weather" = 6 (seems positive)
- Right: "Nice weather" = 2-3 (background detail unless weather-dependent goal)

---

## Validation Checklist

Before finalizing seed dataset:

- [ ] All scores 1-10 represented (min 2 examples each)
- [ ] Diverse agent personalities covered (social, analytical, organized, impulsive, detail-oriented)
- [ ] All observation categories included (social, environmental, goal-relevant, emotional, mundane)
- [ ] Edge cases and ambiguous examples included
- [ ] Each seed has clear rationale
- [ ] Inter-rater agreement checked (if applicable)

---

## Example Seed Entries (Full Context)

### Good Example 1: Clear High Score
```json
{
  "observation": "Fire alarm going off in the building!",
  "agent_goal": "Complete research project",
  "agent_personality": "analytical, introverted",
  "gold_score": 10,
  "rationale": "Immediate safety threat requiring evacuation. Overrides all other priorities regardless of goal or personality."
}
```

### Good Example 2: Context-Dependent Medium Score
```json
{
  "observation": "Bob mentioned he's too busy to attend social events",
  "agent_goal": "Build relationships in the neighborhood",
  "agent_personality": "social, optimistic",
  "gold_score": 6,
  "rationale": "Relevant information about potential friend's availability. Worth remembering for future planning but not urgent or actionable today."
}
```

### Good Example 3: Low Score
```json
{
  "observation": "The grass is green and well-maintained",
  "agent_goal": "Document neighborhood stories",
  "agent_personality": "curious, detail-oriented",
  "gold_score": 2,
  "rationale": "Generic environmental detail with no story value. Not actionable or memorable."
}
```

---

## Updates & Revisions

**Version 1.0** (2025-10-11): Initial rubric created for Day 3 seed collection

---

**Next Steps**: Use this guide to rate all seed observations. If inter-rater kappa < 0.6, revise rubric for clarity.
