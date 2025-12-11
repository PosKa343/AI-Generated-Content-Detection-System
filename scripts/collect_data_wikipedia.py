import time
import random
import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def collect_wikipedia_articles(num_articles=500):

    print("="*70)
    print("COLLECTING HUMAN TEXT FROM WIKIPEDIA")
    print("="*70)

    try:
        import wikipediaapi
    except ImportError:
        print("\n Wikipedia package not installed!")
        print("Installing now...")
        import subprocess
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "wikipedia-api"])
        import wikipediaapi

    wiki = wikipediaapi.Wikipedia('AI_Detection_Project/1.0', 'en')

    topics = [
        # Technology & Computing
        'Artificial_intelligence', 'Machine_learning', 'Deep_learning', 'Neural_network',
        'Computer_science', 'Data_science', 'Cloud_computing', 'Quantum_computing',
        'Cybersecurity', 'Blockchain', 'Internet_of_things', 'Robotics', 'Automation',
        '5G', 'Virtual_reality', 'Augmented_reality', 'Cryptocurrency', 'Software_engineering',

        # Science & Nature
        'Climate_change', 'Renewable_energy', 'Solar_energy', 'Wind_power',
        'Biology', 'Chemistry', 'Physics', 'Astronomy', 'Geology', 'Ecology',
        'Genetics', 'Evolution', 'Photosynthesis', 'DNA', 'Protein', 'Cell_(biology)',
        'Quantum_mechanics', 'Theory_of_relativity', 'Big_Bang', 'Black_hole',

        # Health & Medicine
        'Healthcare', 'Medicine', 'Vaccine', 'Antibiotic', 'Cancer', 'Diabetes',
        'Mental_health', 'Nutrition', 'Exercise', 'Public_health', 'Epidemiology',
        'Neuroscience', 'Brain', 'Heart', 'Immune_system', 'COVID-19_pandemic',

        # Education & Society
        'Education', 'University', 'Online_learning', 'Literacy', 'STEM',
        'Psychology', 'Sociology', 'Anthropology', 'Economics', 'Political_science',
        'History', 'Geography', 'Philosophy', 'Ethics', 'Democracy', 'Human_rights',

        # Engineering & Industry
        'Engineering', 'Civil_engineering', 'Mechanical_engineering', 'Electrical_engineering',
        'Architecture', 'Construction', 'Manufacturing', 'Agriculture', 'Biotechnology',
        'Nanotechnology', 'Aerospace', 'Transportation', 'Sustainability',

        # Additional diverse topics
        'Internet', 'World_Wide_Web', 'Social_media', 'Communication', 'Language',
        'Mathematics', 'Statistics', 'Algorithm', 'Programming', 'Database',
        'Space_exploration', 'Mars', 'Moon', 'International_Space_Station',
        'Energy', 'Electricity', 'Battery', 'Fossil_fuel', 'Nuclear_power',
        'Environment', 'Biodiversity', 'Conservation', 'Pollution', 'Ocean'
    ]

    articles = []
    random.shuffle(topics)

    print(f"\nTarget: {num_articles} texts")
    print(f"Searching {len(topics)} Wikipedia topics...\n")

    for i, topic in enumerate(topics):
        if len(articles) >= num_articles:
            break

        if (i + 1) % 10 == 0:
            print(
                f"Progress: Checked {i+1}/{len(topics)} topics, collected {len(articles)} texts")

        try:
            page = wiki.page(topic)

            if not page.exists():
                continue

            text = page.text

            paragraphs = []

            for p in text.split('\n\n'):
                p = p.strip()
                if len(p) > 200:  # Lower threshold
                    paragraphs.append(p)

            if len(paragraphs) < 3:
                for p in text.split('\n'):
                    p = p.strip()
                    if len(p) > 200:  # Lower threshold
                        paragraphs.append(p)

            # Take up to 5 chunks per article
            for paragraph in paragraphs[:5]:
                if len(articles) >= num_articles:
                    break

                # Clean up the text
                clean_text = paragraph.replace('\n', ' ').strip()

                # Accept smaller texts to get more samples
                if len(clean_text) >= 200:  # Lower from 300
                    articles.append({
                        'text_id': f'human_wiki_{len(articles):04d}',
                        'content': clean_text,
                        'label': 0,
                        'source': 'Wikipedia',
                        'title': topic.replace('_', ' '),
                        'url': page.fullurl,
                        'published_at': '',
                        'query': topic,
                        'collection_method': 'wikipedia_api'
                    })

            time.sleep(0.05)

        except Exception as e:
            continue

    df = pd.DataFrame(articles[:num_articles])
    print(f"\n Total Wikipedia texts collected: {len(df)}")

    if len(df) < num_articles:
        print(f" Only got {len(df)} texts (wanted {num_articles})")
        print("This is fine - we'll balance with AI texts")

    return df


def main():

    if len(sys.argv) < 2:
        print(
            "Usage: python3 collect_data_wikipedia.py GEMINI_KEY [num_samples]")
        print("\nThis script uses:")
        print("  - Wikipedia for human text (100% FREE)")
        print("  - Google Gemini for AI text (FREE)")
        sys.exit(1)

    gemini_key = sys.argv[1]
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 500

    print("="*70)
    print("DATA COLLECTION WITH WIKIPEDIA + GEMINI")
    print("="*70)
    print(
        f"\nCollecting {num_samples} human + {num_samples} AI = {num_samples*2} total")
    print("Using Wikipedia (no NewsAPI needed!)")

    # Collect human text from Wikipedia
    df_human = collect_wikipedia_articles(num_samples)

    if len(df_human) < num_samples:
        print(
            f"\n Only got {len(df_human)} Wikipedia texts (wanted {num_samples})")
        print("That's okay! We'll balance with AI texts.")

    # Save intermediate
    os.makedirs('data/raw', exist_ok=True)
    df_human.to_csv('data/raw/human_wikipedia_articles.csv', index=False)
    print(f"\n Saved: data/raw/human_wikipedia_articles.csv")

    # Collect AI text from Gemini
    print("\n" + "="*70)
    print("COLLECTING AI TEXT FROM GEMINI")
    print("="*70)

    from data_collection import DataCollector
    collector = DataCollector(gemini_key=gemini_key)
    df_ai = collector.generate_ai_text_gemini(num_samples=num_samples)

    # Save intermediate
    df_ai.to_csv('data/raw/ai_gemini_texts.csv', index=False)
    print(f"\n Saved: data/raw/ai_gemini_texts.csv")

    # Create balanced dataset
    print("\n" + "="*70)
    print("CREATING BALANCED DATASET")
    print("="*70)

    # Use the smaller count to balance
    min_samples = min(len(df_human), len(df_ai))

    df_human_balanced = df_human.sample(n=min_samples, random_state=42)
    df_ai_balanced = df_ai.sample(n=min_samples, random_state=42)

    df_final = pd.concat(
        [df_human_balanced, df_ai_balanced], ignore_index=True)
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save final
    df_final.to_csv('data/raw/ai_detection_dataset.csv', index=False)

    print(f"\n Final dataset created!")
    print(f"   Total: {len(df_final)} samples")
    print(f"   Human: {len(df_final[df_final['label']==0])}")
    print(f"   AI: {len(df_final[df_final['label']==1])}")
    print(f"\n Saved: data/raw/ai_detection_dataset.csv")
    print("\n Ready to use in your notebook!")


if __name__ == '__main__':
    main()
