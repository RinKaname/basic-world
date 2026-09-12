import random
import torch
import sys
import os

# Add parent directory to path so we can import model and tokenizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer import BasicWordTokenizer
from model import AmadeusRNN

class Entity:
    def __init__(self, name, is_player=False):
        self.name = name
        self.is_player = is_player
        self.health = 100
        self.hunger = 0   # 0 is full, 100 is starving
        self.thirst = 0   # 0 is quenched, 100 is dehydrated
        self.bag = []     # Inventory
        self.location = "Camp" # Start at Camp
        self.is_alive = True

    def update_stats(self):
        if not self.is_alive: return
        self.hunger += 5
        self.thirst += 8
        if self.hunger >= 100:
            self.hunger = 100
            self.health -= 10
        if self.thirst >= 100:
            self.thirst = 100
            self.health -= 15

        if self.health <= 0:
            self.health = 0
            self.is_alive = False

    def get_status(self):
        if not self.is_alive: return f"{self.name} Is Dead."
        status = [f"{self.name} See {self.location}."]
        if self.hunger > 70: status.append(f"{self.name} Am Hungry." if self.name == "I" else f"{self.name} Is Hungry.")
        if self.thirst > 70: status.append(f"{self.name} Am Thirsty." if self.name == "I" else f"{self.name} Is Thirsty.")
        if self.health < 30: status.append(f"{self.name} Am Hurt." if self.name == "I" else f"{self.name} Is Hurt.")
        return " ".join(status)

class Zone:
    def __init__(self, name, resources):
        self.name = name
        self.resources = resources # dict of {item_name: chance_to_find}

class World:
    def __init__(self):
        self.zones = {
            "Forest": Zone("Forest", {"Apple": 0.6, "Deer": 0.4, "Wood": 0.5}),
            "River": Zone("River", {"Water": 1.0, "Fish": 0.3, "Rock": 0.7}),
            "Camp": Zone("Camp", {"Fire": 1.0, "Rock": 0.4, "Wood": 0.3})
        }
        self.entities = []

    def add_entity(self, entity):
        self.entities.append(entity)

    def step(self):
        # Update the world state for a turn
        for entity in self.entities:
            entity.update_stats()

    def parse_action(self, entity, action_str):
        if not entity.is_alive: return "Dead."

        words = action_str.lower().replace(".", "").split()
        if not words: return "No Action."

        # Action mappings based on basic vocabulary
        # Syntax expected: [Pronoun] [Verb] [Noun]
        if len(words) < 2: return "Bad Action."

        verb = words[1]
        target = words[2].capitalize() if len(words) > 2 else None

        # Support multi-word targets (e.g. "Deer Meat" or "Apple Fruit")
        if len(words) > 2:
            target = " ".join(words[2:]).title()

        if verb == "walk" and target in self.zones:
            entity.location = target
            return f"{entity.name} Walk {target}."

        elif verb == "take" and target:
            zone = self.zones[entity.location]
            if target in zone.resources and random.random() <= zone.resources[target]:
                entity.bag.append(target)
                return f"{entity.name} Take {target}."
            else:
                return f"No {target} In {entity.location}."

        elif verb == "craft" and target == "Axe":
            if "Wood" in entity.bag and "Rock" in entity.bag:
                entity.bag.remove("Wood")
                entity.bag.remove("Rock")
                entity.bag.append("Axe")
                return f"{entity.name} Craft Axe."
            return "No Wood Or Rock."

        elif verb == "chop" and target == "Wood":
            if entity.location == "Forest" and "Axe" in entity.bag:
                entity.bag.append("Wood")
                return f"{entity.name} Chop Wood."
            return "No Axe Or Not In Forest."

        elif verb == "cook" and target == "Meat":
            if entity.location == "Camp":
                # Can cook either generic meat or specific deer/cow meat
                meat_types = ["Meat", "Deer Meat", "Cow Meat"]
                has_meat = any(m in entity.bag for m in meat_types)
                if has_meat and "Wood" in entity.bag:
                    # Remove the first meat found
                    for m in meat_types:
                        if m in entity.bag:
                            entity.bag.remove(m)
                            break
                    entity.bag.remove("Wood")
                    entity.bag.append("Cooked Meat")
                    return f"{entity.name} Cook Meat."
                return "No Meat Or Wood."
            return "Not At Camp."

        elif verb == "eat" and target:
            if target in entity.bag:
                entity.bag.remove(target)
                # Differentiate raw vs cooked meat nutrition
                if target in ["Meat", "Deer Meat", "Cow Meat"]:
                    entity.hunger = max(0, entity.hunger - 20)  # Raw meat restores less
                elif target == "Cooked Meat":
                    entity.hunger = max(0, entity.hunger - 60)  # Cooked meat restores a lot
                else:
                    entity.hunger = max(0, entity.hunger - 40)  # Fruits
                return f"{entity.name} Eat {target}."
            else:
                return f"No {target} In Bag."

        elif verb == "drink" and target:
            if entity.location == "River" and target == "Water":
                entity.thirst = max(0, entity.thirst - 60)
                return f"{entity.name} Drink Water."
            return "Cannot Drink."

        elif verb == "hunt" and target == "Deer":
            if entity.location == "Forest" and random.random() < 0.5:
                entity.bag.append("Deer Meat")
                return f"{entity.name} Hunt Deer. Take Deer Meat."
            return "No Deer."

        elif verb == "sleep":
            entity.health = min(100, entity.health + 20)
            return f"{entity.name} Sleep."

        return "Bad Action."

def run_game(mode="player"):
    # Load Model and Tokenizer
    print("Loading AI Model...")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BasicWordTokenizer()
    # Note: Using relative path from root if run from root, or adjust if run from text/
    import os
    if os.path.exists("text/world_word.txt"):
        vocab_path = "text/world_word.txt"
        model_path = "basic_world_model.safetensors"
    else:
        vocab_path = "world_word.txt"
        model_path = "../basic_world_model.safetensors"

    tokenizer.build_vocab(vocab_path)
    model = AmadeusRNN(vocab_size=tokenizer.vocab_size, hidden_dim=64).to(DEVICE)

    from safetensors.torch import load_file
    try:
        model.load_state_dict(load_file(model_path))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Could not load model: {e}. Falling back to random bot.")
        model = None

    world = World()

    if mode == "player":
        player = Entity("I", is_player=True)
        bot = Entity("He", is_player=False)
        world.add_entity(player)
        world.add_entity(bot)
        print("=== Basic World: Player Co-Op ===")
    else:
        bot1 = Entity("He", is_player=False)
        bot2 = Entity("She", is_player=False)
        world.add_entity(bot1)
        world.add_entity(bot2)
        print("=== Basic World: Agent Only ===")

    print("Commands: [I Walk Forest], [I Take Apple], [I Eat Apple], [I Drink Water], [I Hunt Deer]")

    turn = 1
    while True:
        print(f"\n--- Turn {turn} ---")

        for entity in world.entities:
            if not entity.is_alive:
                continue

            print(f"[{entity.name}] Status: HP:{entity.health} Hunger:{entity.hunger} Thirst:{entity.thirst} Bag:{entity.bag}")
            print(entity.get_status())

            if entity.is_player:
                action = input(f"Action for {entity.name}: ")
                if action.lower() == "stop": return
            else:
                if model is not None:
                    # Let the SLM decide based on current state (prompt)
                    prompt = entity.get_status() + f" {entity.name}"
                    encoded = tokenizer.encode(prompt)
                    input_tensor = torch.tensor(encoded, dtype=torch.long, device=DEVICE).unsqueeze(0)

                    with torch.no_grad():
                        # Generate up to 3 words (Verb + Target Word 1 + Target Word 2) to support "Deer Meat"
                        gen_tokens = model.generate(input_tensor, max_new_tokens=3, temperature=0.8)

                    # Decode only the newly generated tokens
                    generated_ids = gen_tokens[0].cpu().numpy().tolist()[len(encoded):]
                    generated_words = [tokenizer.id2word.get(tid, "") for tid in generated_ids if tid != 0]

                    action = f"{entity.name} " + " ".join(generated_words).title()
                    original_action_raw = action

                    # Clean the action text from periods and extra tokens
                    clean_action = action.replace(".", "").strip()
                    words = clean_action.lower().split()

                    # Truncate if the model predicts the start of a new sentence (e.g. "He Walk Forest He")
                    if len(words) > 2 and words[-1] in ["he", "she", "it", "they", "we", "you", "i"]:
                        words = words[:-1]

                    # Fix hallucination like "He Walk He Stop" -> "He Walk Forest"
                    if len(words) > 2 and words[2] in ["he", "she", "i", "stop", "see"]:
                        words = words[:2]

                    clean_action = " ".join(words).title()

                    fallback_triggered = False
                    # Sanity check: fallback if model hallucinated un-parsable garbage
                    if len(words) < 2 or words[1] not in ["walk", "take", "eat", "drink", "hunt", "sleep", "craft", "chop", "cook"]:
                         fallback_triggered = True
                         clean_action = f"{entity.name} Walk {random.choice(['Forest', 'River', 'Camp'])}"

                    # Secondary fallback if the target is missing (e.g. just "He Walk")
                    elif len(words) == 2 and words[1] == "walk":
                         fallback_triggered = True
                         clean_action = f"{entity.name} Walk {random.choice(['Forest', 'River', 'Camp'])}"

                    action = clean_action

                    if fallback_triggered:
                        print(f"[SLM Hallucination] Original output: '{original_action_raw}'. Overriding with fallback.")
                else:
                    # Basic Random fallback AI
                    possible_actions = [f"{entity.name} Walk Forest", f"{entity.name} Walk River", f"{entity.name} Walk Camp"]
                    if entity.location == "Forest":
                        possible_actions.extend([f"{entity.name} Take Apple", f"{entity.name} Hunt Deer"])
                    if entity.location == "River":
                        possible_actions.append(f"{entity.name} Drink Water")
                    if entity.bag:
                        possible_actions.append(f"{entity.name} Eat {entity.bag[0]}")

                    action = random.choice(possible_actions)

                print(f"Agent {entity.name} chooses: {action}")

            result = world.parse_action(entity, action)
            print(f"> {result}")

        world.step()
        turn += 1

        if not any(e.is_alive for e in world.entities):
            print("All Dead. Game Over.")
            break

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        print("Select game mode:")
        print("1. Player Co-Op (You + AI Agent)")
        print("2. Agent Only (2 AI Agents)")
        choice = input("Enter 1 or 2: ").strip()

        mode = "player" if choice == "1" else "agent"

    run_game(mode)
