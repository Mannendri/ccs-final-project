# How People Schedule (and Avoid) Tasks: A Study on Prioritization and Procrastination

**Team:** Tori Kelley, Brios Olivares, Mannendri Olivares  
**Contact TA:** Yoni Friedman

## Core Idea

People tend to avoid tasks that make them feel unintelligent or uncertain, steering away from activities that are abstract, difficult, or cognitively demanding.

## Intuition

Humans often resist tasks that feel confusing, effortful, or likely to expose a lack of understanding. As a result, they may gravitate toward simpler, more concrete activities that feel easier to complete or measure.

## Research Question

How do people decide which tasks to tackle first and which to de-prioritize? What types of tasks are most likely to be procrastinated?

## Hypothesis

We hypothesize that individual differences in procrastination tendencies will be partly explained by differences in task representation. Specifically, participants who mentally represent tasks as more concrete, predictable, and rewarding will prioritize them earlier, whereas those who represent the same tasks as abstract, uncertain, or effortful will be more likely to delay them.

## Research Goals

- Build a computational model of human task prioritization and evaluation to anticipate biases in how people schedule and select tasks.
- Test how people strategically approach or avoid certain tasks based on perceived qualities such as difficulty, abstraction, and expected reward.

## Set-Up

We will solicit volunteers from Mannendri and Brios' fraternity to participate in a week-long study. Since members are already required to complete a weekly chore (a task that is often procrastinated), we will adapt this system by adding several additional small tasks to track throughout the week.

Participants will be given a list of tasks to both schedule and complete within a 1-week window from December 1 (Monday) to December 7 (Sunday). These tasks vary along the following hidden dimensions:

- Concreteness
- Difficulty
- Duration

Participants will complete eight tasks that systematically vary across these dimensions. These include:

- sorting a deck of playing cards
- completing a short quiz
- solving a logic puzzle
- completing their assigned fraternity chore
- writing a brief reflection
- draw for ten minutes
- explaining a difficult concept from class
- analyzing a short reading

| Task | Deadline | Concreteness | Difficulty | Duration | Point value |
|------|----------|--------------|------------|----------|-------------|
| Sort a deck of cards | Tuesday | high | easy | short | 5 pts |
| Complete a short quiz | Thursday | high | easy | short | 5 pts |
| Complete fraternity chore | Saturday | high | medium | long | 25 pts |
| Write a brief reflection | Tuesday | medium | medium | medium | 15 pts |
| Draw for ten minutes | Thursday | medium | easy | medium | 10 pts |
| Solve a logic puzzle | Saturday | medium | hard | medium | 15 pts |
| Explain a difficult concept from class | Tuesday | low | hard | medium | 20 pts |
| Analyze a short passage | Thursday | low | hard | long | 20 pts |

These tasks range from simple and concrete to open-ended and cognitively demanding, allowing us to measure how task characteristics influence scheduling and procrastination behavior. Although all task submissions will be accepted through the final end-of-week deadline, each task will have an additional deadline throughout the week (one of Tuesday, Thursday, or Saturday), and a temporal discount of 10% will be subtracted from the points awarded for completing a task each day late.

### Planning Phase

At the beginning of the week, participants will rate tasks on perceived difficulty, abstraction, and estimated duration, so we can use their perception of the task to model when they schedule it.

### Completion Logging

Participants will log task completion using a timestamped self-report form. If a task is completed in multiple sessions, multiple check-ins will capture this splitting behavior.

## Analysis and Modeling

We will compare human behavior against:

- Random assignment
- Optimal/utility-maximizing assignment
- Standard temporal discounting models, which assume rewards lose value as deadlines get farther away.

To capture procrastination driven by task properties (not just timing), we will build a simple generative model using [memo](https://github.com/kach/memo). In "Self-Confidence and Personal Motivation" a general cognitive model is as follows:

```
Utility(task) = expected payoff − expected drop in self belief
```

We hope to build upon this model by considering the effect of temporal discount with task deadlines, in addition to exploring how perceived abstraction and difficulty of a task inform the participant's expected drop in self-belief. The model will assume that the subjective utility of choosing task j includes both:

- Discounted reward (as in temporal discounting), and
- Costs associated with abstraction, difficulty, and uncertainty (cognitive aversion).

We will infer per-participant parameters for both components and compare predictive accuracy across models. This will enable us to determine whether temporal discounting alone can account for behavior, or whether cognitive aversion is also required.

## Expected Outcomes / Possible Implications

Participants will likely show a preference toward scheduling concrete, low-uncertainty tasks first, even when abstract tasks could yield higher rewards. This behavior could reveal the underlying heuristics people use when faced with uncertain effort or ambiguous payoffs for tasks with deadlines.
