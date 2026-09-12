import random

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
            "Forest": Zone("Forest", {"Apple": 0.6, "Deer": 0.4, "Wood": 0.8}),
            "River": Zone("River", {"Water": 1.0, "Fish": 0.3}),
            "Camp": Zone("Camp", {"Fire": 1.0})
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

        elif verb == "eat" and target:
            if target in entity.bag:
                entity.bag.remove(target)
                entity.hunger = max(0, entity.hunger - 40)
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
                entity.bag.append("Meat")
                return f"{entity.name} Hunt Deer. Take Meat."
            return "No Deer."

        elif verb == "sleep":
            entity.health = min(100, entity.health + 20)
            return f"{entity.name} Sleep."

        return "Bad Action."

def run_game(mode="player"):
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
                # Basic Bot AI - pick random valid action based on state
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
