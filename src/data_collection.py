import google.generativeai as genai
import requests
import pandas as pd
import time
import random
from datetime import datetime, timedelta
from typing import List, Optional
import os


class DataCollector:

    def __init__(self, newsapi_key: str = None, gemini_key: str = None):

        self.newsapi_key = newsapi_key or os.getenv('NEWSAPI_KEY')
        self.gemini_key = gemini_key or os.getenv('GEMINI_KEY')

        if self.gemini_key:
            genai.configure(api_key=self.gemini_key)
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE"},
            ]
            self.gemini_model = genai.GenerativeModel(
                'gemini-flash-latest',
                safety_settings=safety_settings
            )
        else:
            self.gemini_model = None

    def collect_news_articles(self,
                              num_articles: int = 500,
                              queries: List[str] = None,
                              days_back: int = 30) -> pd.DataFrame:

        if not self.newsapi_key:
            raise ValueError("NewsAPI key not provided.")

        print("="*70)
        print("COLLECTING HUMAN-WRITTEN NEWS ARTICLES")
        print("="*70)

        if queries is None:
            queries = [
                'technology',
                'artificial intelligence',
                'climate change',
                'education',
                'healthcare',
                'science',
                'innovation',
                'research'
            ]

        base_url = 'https://newsapi.org/v2/everything'
        articles = []
        articles_per_query = num_articles // len(queries)

        for query in queries:
            print(f"\nSearching for: {query}")

            params = {
                'q': query,
                'apiKey': self.newsapi_key,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 100
            }

            try:
                response = requests.get(base_url, params=params)
                data = response.json()

                if data.get('status') != 'ok':
                    print(f"   Error: {data.get('message', 'Unknown error')}")
                    continue

                # Extract articles
                query_articles = 0
                for article in data.get('articles', []):
                    content = article.get('content', '')
                    description = article.get('description', '')
                    title = article.get('title', '')

                    full_text = f"{title}. {description} {content}"

                    if len(full_text) > 300:
                        articles.append({
                            'text_id': f'human_{len(articles):04d}',
                            'content': full_text,
                            'label': 0,
                            'source': article.get('source', {}).get('name', 'unknown'),
                            'title': title,
                            'url': article.get('url', ''),
                            'published_at': article.get('publishedAt', ''),
                            'query': query,
                            'collection_method': 'newsapi'
                        })
                        query_articles += 1

                    if query_articles >= articles_per_query:
                        break

                print(f"   Collected {query_articles} articles")
                time.sleep(1)

            except Exception as e:
                print(f"   Error collecting {query}: {e}")
                continue

            if len(articles) >= num_articles:
                break

        df = pd.DataFrame(articles[:num_articles])
        print(f"\n Total human articles collected: {len(df)}")

        return df

    def generate_ai_text_gemini(self,
                                num_samples: int = 500,
                                topics: List[str] = None) -> pd.DataFrame:

        if not self.gemini_model:
            raise ValueError("Gemini API key not provided")

        print("\n" + "="*70)
        print("GENERATING AI TEXT WITH GOOGLE GEMINI")
        print("="*70)

        if topics is None:
            topics = [
                'artificial intelligence', 'machine learning', 'climate change',
                'renewable energy', 'technology', 'education', 'healthcare',
                'space exploration', 'cybersecurity', 'data science',
                'biotechnology', 'social media', 'automation', 'robotics'
            ]

        prompt_templates = [
            "Write a {length}-word informative article about {topic} and its impact on society.",
            "Explain the importance of {topic} in modern times. Write about {length} words.",
            "Discuss the future of {topic} and provide detailed analysis. Around {length} words.",
            "Analyze how {topic} is changing our world. Write a {length}-word essay.",
            "Describe the benefits and challenges of {topic}. Write approximately {length} words.",
            "Write an educational piece about {topic} for a general audience. {length} words.",
            "Explore the role of {topic} in addressing global challenges. {length} words.",
            "Provide a comprehensive overview of {topic} and its applications. {length} words."
        ]

        ai_texts = []
        errors = 0
        successful = 0
        attempts = 0

        while successful < num_samples and attempts < num_samples * 2:
            topic = random.choice(topics)
            template = random.choice(prompt_templates)
            length = random.choice([300, 400, 500])

            prompt = template.format(topic=topic, length=length)

            try:
                response = self.gemini_model.generate_content(prompt)

                if not response or not response.text or len(response.text.strip()) < 100:
                    print(
                        f"   Empty or short response on attempt {attempts}, retrying...")
                    attempts += 1
                    time.sleep(1)
                    continue

                ai_texts.append({
                    'text_id': f'ai_gemini_{successful:04d}',
                    'content': response.text,
                    'label': 1,
                    'source': 'google_gemini_flash_latest',
                    'prompt': prompt,
                    'topic': topic,
                    'collection_method': 'gemini_api'
                })

                successful += 1
                attempts += 1

                # Progress update
                if successful % 50 == 0:
                    print(
                        f"  Generated {successful}/{num_samples} (attempts: {attempts})")

                time.sleep(1.1)

            except Exception as e:
                errors += 1
                attempts += 1

                error_msg = str(e)

                if "block" in error_msg.lower() or "safety" in error_msg.lower():
                    print(
                        f"   Content blocked on attempt {attempts}, trying different topic...")
                    time.sleep(1)
                    continue

                print(f"   Error on attempt {attempts}: {e}")

                # If rate limited, wait longer
                if "429" in error_msg or "quota" in error_msg.lower():
                    print("  ⏸ Rate limit hit, waiting 60 seconds...")
                    time.sleep(60)
                elif errors > 20:
                    print(
                        f"   Too many errors ({errors}), stopping generation")
                    break

                time.sleep(2)
                continue

        df = pd.DataFrame(ai_texts)
        print(f"\n Total AI texts generated: {len(df)}")
        print(
            f"  Successful: {successful}, Attempts: {attempts}, Errors: {errors}")

        if len(df) < num_samples * 0.8:
            print(
                f"\n WARNING: Only got {len(df)} samples (wanted {num_samples})")
            print("This might be due to:")
            print("  - Safety filters blocking content")
            print("  - Rate limiting")
            print("  - API issues")
            print("\nYou can:")
            print("  1. Run the script again to collect more")
            print("  2. Use what you have (still usable!)")
            print("  3. Try different topics")

        return df

    def create_balanced_dataset(self,
                                df_human: pd.DataFrame,
                                df_ai: pd.DataFrame,
                                min_length: int = 300,
                                max_length: int = 2000) -> pd.DataFrame:

        print("\n" + "="*70)
        print("CREATING BALANCED DATASET")
        print("="*70)

        # Combine datasets
        df_combined = pd.concat([df_human, df_ai], ignore_index=True)
        print(f"\nInitial total samples: {len(df_combined)}")

        print("\nCleaning data...")

        df_combined = df_combined.dropna(subset=['content'])
        print(f"  After removing NaN: {len(df_combined)}")

        df_combined = df_combined.drop_duplicates(subset=['content'])
        print(f"  After removing duplicates: {len(df_combined)}")

        df_combined = df_combined[
            (df_combined['content'].str.len() >= min_length) &
            (df_combined['content'].str.len() <= max_length)
        ]
        print(
            f"  After length filter ({min_length}-{max_length} chars): {len(df_combined)}")

        n_human = len(df_combined[df_combined['label'] == 0])
        n_ai = len(df_combined[df_combined['label'] == 1])
        min_samples = min(n_human, n_ai)

        print(f"\nBalancing classes:")
        print(f"  Human samples: {n_human}")
        print(f"  AI samples: {n_ai}")
        print(f"  Using {min_samples} samples per class")

        df_human_balanced = df_combined[df_combined['label'] == 0].sample(
            n=min_samples, random_state=42
        )
        df_ai_balanced = df_combined[df_combined['label'] == 1].sample(
            n=min_samples, random_state=42
        )

        df_final = pd.concat(
            [df_human_balanced, df_ai_balanced], ignore_index=True)

        df_final = df_final.sample(
            frac=1, random_state=42).reset_index(drop=True)

        df_final['collection_date'] = datetime.now().strftime('%Y-%m-%d')

        print(f"\n Final balanced dataset: {len(df_final)} samples")

        return df_final

    def collect_full_dataset(self,
                             num_samples: int = 500,
                             save_intermediates: bool = True,
                             output_dir: str = '../data/raw') -> pd.DataFrame:

        print("\n" + "="*70)
        print("AI DETECTION DATA COLLECTION PIPELINE")
        print("Using: Google Gemini + NewsAPI")
        print("="*70)

        os.makedirs(output_dir, exist_ok=True)

        df_human = self.collect_news_articles(num_articles=num_samples)

        if save_intermediates:
            human_path = os.path.join(output_dir, 'human_news_articles.csv')
            df_human.to_csv(human_path, index=False)
            print(f"\n Saved human articles to: {human_path}")

        df_ai = self.generate_ai_text_gemini(num_samples=num_samples)

        if save_intermediates:
            ai_path = os.path.join(output_dir, 'ai_gemini_texts.csv')
            df_ai.to_csv(ai_path, index=False)
            print(f"\n Saved AI texts to: {ai_path}")

        df_final = self.create_balanced_dataset(df_human, df_ai)

        final_path = os.path.join(output_dir, 'ai_detection_dataset.csv')
        df_final.to_csv(final_path, index=False)

        print("\n" + "="*70)
        print("DATASET CREATION COMPLETE!")
        print("="*70)
        print(f"\nFinal Statistics:")
        print(f"  Total samples: {len(df_final)}")
        print(f"  Human samples: {len(df_final[df_final['label'] == 0])}")
        print(f"  AI samples: {len(df_final[df_final['label'] == 1])}")
        print(
            f"  Average length: {df_final['content'].str.len().mean():.0f} characters")
        print(f"\nSaved to: {final_path}")

        # Show samples
        print("\n" + "="*70)
        print("SAMPLE DATA:")
        print("="*70)
        print("\nHuman sample:")
        print(df_final[df_final['label'] == 0].iloc[0]
              ['content'][:200] + "...")
        print("\nAI sample:")
        print(df_final[df_final['label'] == 1].iloc[0]
              ['content'][:200] + "...")

        return df_final


def collect_data(newsapi_key: str,
                 gemini_key: str,
                 num_samples: int = 500,
                 output_dir: str = '../data/raw') -> pd.DataFrame:

    collector = DataCollector(newsapi_key=newsapi_key, gemini_key=gemini_key)
    return collector.collect_full_dataset(
        num_samples=num_samples,
        save_intermediates=True,
        output_dir=output_dir
    )
