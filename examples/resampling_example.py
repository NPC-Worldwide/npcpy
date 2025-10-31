import os
import random
from collections import Counter
import csv # We need this for CSV output!
from npcpy.npc_compiler import NPC
from npcpy.llm_funcs import get_llm_response

# Ensure you have Ollama running and the model available, e.g., 'ollama run gemma3:4b'

def generate_random_persona() -> dict:
    """
    Generates a random NPC persona with a unique name, role, adjective, place, and time.
    The primary directive focuses on the sentiment task, with the persona influencing the interpretation.
    """
    base_names = ["Sofia", "Giovanni", "Isabella", "Marco", "Giulia", "Luca", "Elena", "Antonio", "Caterina", "Francesco", "Pietro", "Rosa"]
    roles = ["historian", "scientist", "poet", "chef", "traveler", "philosopher", "artist", "merchant", "strategist", "botanist", "engineer", "musician"]
    adjectives = ["curious", "cynical", "enthusiastic", "stoic", "witty", "melancholy", "observant", "bold", "pragmatic", "dreamy", "analytical", "passionate"]
    places = [
        "ancient Rome", "a bustling market in Marrakesh", "a futuristic space station",
        "a quiet monastery in Tibet", "a pirate ship in the Caribbean", "the court of Louis XIV",
        "a remote research outpost in Antarctica", "a vibrant jazz club in 1920s New York",
        "a medieval castle in Scotland", "a bustling port in ancient Alexandria",
        "a hidden village in the Amazon rainforest", "a grand opera house in Vienna"
    ]
    times = [
        "the 18th century", "the year 3000", "the medieval era", "the Renaissance",
        "the roaring twenties", "the age of discovery", "the dawn of the internet",
        "a time before recorded history", "the Victorian era", "the space age",
        "the Industrial Revolution", "the Baroque period"
    ]

    base_name = random.choice(base_names)
    unique_id = random.randint(100, 999)
    name = f"{base_name}_{unique_id}"
    
    role = random.choice(roles)
    adjective = random.choice(adjectives)
    place = random.choice(places)
    time = random.choice(times)

    primary_directive = (
        f"You are {adjective} {name}, a {role} from {place} in {time}. "
        f"Your task is to analyze the provided text and classify its sentiment as 'positive', 'negative', or 'neutral'. "
        f"Respond only with the single classification word, nothing else. "
        f"Your unique background and perspective should subtly influence your interpretation of the sentiment."
    )

    return {
        "name": name,
        "primary_directive": primary_directive,
        "description": f"{adjective} {role} from {place} in {time}",
        "role": role, # Include these for more granular analysis in CSV
        "adjective": adjective,
        "place": place,
        "time": time
    }


def get_sentiment_from_new_persona(text: str) -> dict:
    """
    Generates a new persona, creates an NPC, and gets a single sentiment classification from it.
    Returns a dictionary with persona details and the classification.
    """
    persona_data = generate_random_persona()
    dynamic_npc = NPC(
        name=persona_data["name"],
        primary_directive=persona_data["primary_directive"],
        model='gemma3:4b',  # Using a local Ollama model
        provider='ollama'
        # model='gemini-1.5-flash', # Or use a remote model if you have the API key
        # provider='gemini'
    )
    
    response_obj = get_llm_response(text, npc=dynamic_npc)
    classification = response_obj['response'].strip().lower()
    
    result = {
        "original_text": text,
        "persona_name": persona_data["name"],
        "persona_description": persona_data["description"],
        "persona_role": persona_data["role"],
        "persona_adjective": persona_data["adjective"],
        "persona_place": persona_data["place"],
        "persona_time": persona_data["time"],
        "sentiment_classification": classification
    }
    return result

if __name__ == "__main__":
    texts_to_classify = [
        "I absolutely love this new limoncello recipe, it's a burst of sunshine!",
        "The recent eruption was a bit too close for comfort, quite alarming.",
        "Mount Etna is a volcano in Sicily.",
        "This pasta is just okay.",
        "Please dont die alone there isnt a chance that these roofs were all made in the same day but hypothetically speaking it is possible but why should we subject ourselves to the machinations of the lake the lake that supposedly gives us life but damn if i havent seen more people taken by the lake than people fed full for life by that same damn place.",
        "I despise cold weather and northerners.",
        "HE LONGED FOR A DAY WHEN BOXES WOULD BE MADE OF LIZARDS AGAIN AND NOT THIS PUDGY PLASTIC YUCK."
    ]

    NUM_PERSONAS_TO_SAMPLE = 25 
    OUTPUT_CSV_FILENAME = "sentiment_analysis_results.csv"

    print("\n--- Starting Sentiment Classification with Fresh Persona Sampling for Each Opinion ---\n")

    all_results_for_csv = [] # This list will hold all dictionaries for CSV output

    for i, text in enumerate(texts_to_classify):
        print(f"\n--- Processing Text {i+1}: '{text}' ---")
        
        current_text_classifications = [] # For printing the distribution for current text
        print(f"🌋 Collecting {NUM_PERSONAS_TO_SAMPLE} sentiment classifications for '{text}' from unique personas...")

        for _ in range(NUM_PERSONAS_TO_SAMPLE):
            result = get_sentiment_from_new_persona(text)
            all_results_for_csv.append(result) # Add to the master list for CSV
            current_text_classifications.append(result["sentiment_classification"]) # For current text's distribution
        
        # Calculate and print the distribution for the current text (as before)
        classification_distribution = Counter(current_text_classifications)
        
        print(f"\n✨ Distribution of sentiment classifications across {NUM_PERSONAS_TO_SAMPLE} unique personas for '{text}':")
        for sentiment, count in classification_distribution.items():
            print(f"  - {sentiment.upper()}: {count} ({(count / NUM_PERSONAS_TO_SAMPLE) * 100:.1f}%)")
        print("-" * 50)

    # Now, write all collected results to a CSV file!
    if all_results_for_csv:
        # Get headers from the first dictionary
        csv_headers = all_results_for_csv[0].keys()
        
        with open(OUTPUT_CSV_FILENAME, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
            writer.writeheader()
            writer.writerows(all_results_for_csv)
        print(f"\n🔥 All sentiment analysis results saved to '{OUTPUT_CSV_FILENAME}' for your deep dive! Enjoy the data flow!")
    else:
        print("\nNo results to save to CSV.")

    print("\n--- Diverse Sentiment Classification Complete! ---")