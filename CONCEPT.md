# Conceptual Design: Basic English Simulation World

This document outlines the high-level concepts for creating a simulated world where AI agents communicate using a very basic, restricted form of English. The goal is to keep the compute requirements low while enabling emergent behavior and interaction.

## 1. The Core Concept
The idea is to build a simulated environment (either text-based or a simple 2D grid) populated by AI agents. Instead of using massive, state-of-the-art Large Language Models (LLMs) with billions of parameters, these agents are powered by extremely small, custom-trained language models. The key restriction is that these agents only know and understand "Basic English"—a highly constrained vocabulary of essential words.

## 2. Model Size and Architecture
*   **Small Models:** You absolutely do not need a billion or trillion-parameter model for this. Models with a few million parameters (e.g., small RNNs, LSTMs, or tiny Transformer models) are more than capable of generating coherent short sentences and making decisions based on restricted inputs.
*   **Training:** Instead of training on the entire internet, the model would be trained exclusively on a custom dataset of basic interactions, actions, and environmental descriptions.

## 3. Constrained Vocabulary & Tokenization
*   **The Dictionary:** The simulation would define a strict dictionary (e.g., 100 to 500 words). Words would include basic verbs (go, eat, take, give, talk), nouns (apple, tree, rock, water, human), and simple modifiers (big, small, good, bad).
*   **Tokenization:** Because the vocabulary is so small, tokenization becomes trivial. Each word in the basic dictionary can map to a single token. This vastly reduces the complexity of the embedding layer and the model's output layer. The vocabulary size (and therefore the softmax output size) is hundreds, not tens of thousands like in standard LLMs.
*   **Grammar:** Grammar can be simplified. "I want apple" instead of "I would like an apple."

## 4. The Simulation Environment
The environment acts as the prompt/context for the agents and parses their output.

### Option A: Text-Based (MUD Style)
*   The world is represented purely through text descriptions (e.g., "You are in a forest. You see a tree. You see Bob.").
*   Agents output basic text to interact ("take apple from tree", "say hello to Bob").
*   A central engine parses these simple sentences and updates the world state.

### Option B: 2D Grid World (RPG Style)
*   The world is a spatial grid. Agents have x/y coordinates.
*   The agent's "vision" is translated into basic text ("tree at north", "water at south").
*   The agent outputs text that maps to both physical actions ("move north") and dialogue ("talk to Bob: I am hungry").

## 5. Why This Approach Works
*   **Highly Efficient:** Tiny models run very fast on standard CPUs. You could simulate dozens or hundreds of agents simultaneously on a regular computer.
*   **Focused Intent:** By constraining the language, you force the agents to focus on actions and core needs (survival, trading, simple socialization) rather than getting lost in philosophical rambling.
*   **Easier Debugging:** With a 500-word vocabulary, it is much easier to track exactly why an agent made a certain decision or said a certain thing.
