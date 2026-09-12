def append_logic():
    sentences = []
    subjects = ["I", "You", "We", "They", "He", "She", "It"]

    for s in subjects:
        # Survival Logic
        sentences.append(f"{s} Is Thirsty. {s} Walk River. {s} Drink Water.")
        sentences.append(f"{s} Am Thirsty. {s} Walk River. {s} Drink Water." if s == "I" else f"{s} Are Thirsty. {s} Walk River. {s} Drink Water." if s in ["You", "We", "They"] else "")

        sentences.append(f"{s} Is Hungry. {s} Eat Cooked Meat.")
        sentences.append(f"{s} Am Hungry. {s} Eat Cooked Meat." if s == "I" else f"{s} Are Hungry. {s} Eat Cooked Meat." if s in ["You", "We", "They"] else "")

        sentences.append(f"{s} Is Hungry. {s} Eat Apple.")
        sentences.append(f"{s} Am Hungry. {s} Eat Apple." if s == "I" else f"{s} Are Hungry. {s} Eat Apple." if s in ["You", "We", "They"] else "")

        sentences.append(f"{s} Is Hurt. {s} Sleep.")
        sentences.append(f"{s} Am Hurt. {s} Sleep." if s == "I" else f"{s} Are Hurt. {s} Sleep." if s in ["You", "We", "They"] else "")

        # Crafting Logic
        sentences.append(f"{s} Take Rock. {s} Take Wood. {s} Craft Axe.")
        sentences.append(f"{s} Craft Axe. {s} Walk Forest. {s} Chop Wood.")

        # Cooking Logic
        sentences.append(f"{s} Hunt Deer. {s} Take Deer Meat. {s} Walk Camp. {s} Cook Meat. {s} Eat Cooked Meat.")
        sentences.append(f"{s} See Fire. {s} Cook Meat.")

    # Remove empty strings from ternary ops
    sentences = [seq for seq in sentences if seq]

    # We want these sequences to appear multiple times so the model heavily weights this logic
    weighted_sentences = sentences * 3

    with open("text/world_word.txt", "a") as f:
        for seq in weighted_sentences:
            f.write(seq + "\n")

if __name__ == "__main__":
    append_logic()
    print("Appended logic sequences to dataset.")
