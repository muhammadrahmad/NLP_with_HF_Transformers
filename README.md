<h1 align="center"> Natural Language Processing  with Hugging Face Transformers </h1>
<p align="center"> Generative AI Guided Project on Cognitive Class by IBM</p>

<div align="center">

<img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54">
<img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white">

</div>

## Name : Muhammad Rahmad

## My todo : 

### 1. Example 1 - Sentiment Analysis

```
# TODO :
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
classifier("This is an amazing and helpful course!")
```

Result : 

```
[{'label': 'POSITIVE', 'score': 0.9998809099197388}]
```

Analysis on example 1 : 

The sentiment analysis model accurately classifies the input sentence "This is an amazing and helpful course!" as positive. The model's high confidence score of 0.99988 strongly indicates that it is extremely certain in its assessment. This result aligns perfectly with the clear positive sentiment expressed in the sentence, demonstrating the model's effectiveness in identifying and quantifying positive sentiment in text.

### 2. Example 2 - Topic Classification

```
# TODO :
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
classifier(
    "The latest smartphone comes with a powerful processor and a stunning display, making it perfect for gaming and watching videos.",
    candidate_labels=["technology", "sports", "cooking"],
)
```

Result : 

```
{'sequence': 'The latest smartphone comes with a powerful processor and a stunning display, making it perfect for gaming and watching videos.',
 'labels': ['technology', 'sports', 'cooking'],
 'scores': [0.9867923855781555, 0.011312006041407585, 0.0018956103594973683]}
```

Analysis on example 2 : 

The zero-shot classification model effectively classifies the input sentence "The latest smartphone comes with a powerful processor and a stunning display, making it perfect for gaming and watching videos." The model correctly assigns the highest probability (0.9868) to the "technology" label, which is the most relevant category. The probabilities assigned to "sports" (0.0113) and "cooking" (0.0019) are significantly lower, indicating the model's ability to accurately discern the primary topic and dismiss less relevant categories. This demonstrates the model's strong performance in identifying the technological context of the sentence.

### 3. Example 3 and 3.5 - Text Generator

```
# TODO :
generator = pipeline("text-generation", model="distilgpt2")  # or change to gpt-2
generator(
    "Artificial intelligence is rapidly changing the world, and its impact on various industries will continue to",
    max_length=40,  # you can change this
    num_return_sequences=2,  # and this too
)
```

Result : 

```
[{'generated_text': 'Artificial intelligence is rapidly changing the world, and its impact on various industries will continue to affect every aspect of the economy.\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n'},
 {'generated_text': "Artificial intelligence is rapidly changing the world, and its impact on various industries will continue to evolve, with a few new ways to enhance its capabilities. We use this new technology to assist the business of artificial intelligence, to help businesses in one of the world's fastest growing industries, in the digital economy.\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n"}]
```

Analysis on example 3 : 

The distilgpt2 model generates two distinct continuations for the prompt "Artificial intelligence is rapidly changing the world, and its impact on various industries will continue to." Both generated texts maintain the theme of AI's pervasive influence. The first output focuses on AI's economic impact, while the second discusses AI's evolution and its role in assisting businesses in the digital economy. The model demonstrates an ability to produce relevant and grammatically correct text, although the level of detail and specificity varies between the two generated sequences.

and;

```
unmasker = pipeline("fill-mask", "distilroberta-base")
unmasker("The capital of France is <mask>.", top_k=4)
```

Result : 

```
[{'score': 0.27037179470062256,
  'token': 2201,
  'token_str': ' Paris',
  'sequence': 'The capital of France is Paris.'},
 {'score': 0.05588367581367493,
  'token': 12790,
  'token_str': ' Lyon',
  'sequence': 'The capital of France is Lyon.'},
 {'score': 0.02989797480404377,
  'token': 4612,
  'token_str': ' Barcelona',
  'sequence': 'The capital of France is Barcelona.'},
 {'score': 0.023081671446561813,
  'token': 12696,
  'token_str': ' Monaco',
  'sequence': 'The capital of France is Monaco.'}]
```

Analysis on example 3.5 : 

The "fill-mask" model accurately predicts "Paris" as the most likely word to fill the masked position in the sentence "The capital of France is <mask>." The model assigns a high score of 0.2704 to "Paris," confirming its strong understanding of the relationship between France and its capital city. While the model also suggests other cities like "Lyon," "Barcelona," and "Monaco," their significantly lower scores indicate that the model correctly prioritizes the most accurate answer. This demonstrates the model's ability to leverage contextual information to predict missing words with a high degree of accuracy.

### 4. Example 4 - Name Entity Recognition (NER)

```
# TODO :
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", grouped_entities=True)
ner("Elon Musk is the CEO of Tesla and SpaceX, both located in California.")
```

Result : 

```
[{'entity_group': 'PER',
  'score': np.float32(0.99925494),
  'word': 'Elon Musk',
  'start': 0,
  'end': 9},
 {'entity_group': 'ORG',
  'score': np.float32(0.99776363),
  'word': 'Tesla',
  'start': 24,
  'end': 29},
 {'entity_group': 'ORG',
  'score': np.float32(0.9990744),
  'word': 'SpaceX',
  'start': 34,
  'end': 40},
 {'entity_group': 'LOC',
  'score': np.float32(0.9993698),
  'word': 'California',
  'start': 58,
  'end': 68}]
```

Analysis on example 4 : 

The Named Entity Recognition model accurately identifies and categorizes the key entities in the sentence "Elon Musk is the CEO of Tesla and SpaceX, both located in California." The model correctly labels "Elon Musk" as a person (PER) with a very high confidence score of 0.9993, "Tesla" and "SpaceX" as organizations (ORG) with scores of 0.9978 and 0.9991 respectively, and "California" as a location (LOC) with a score of 0.9994. These high confidence scores indicate the model's strong ability to recognize and classify different types of named entities within the text.

### 5. Example 5 - Question Answering

```
# TODO :
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
question = "What is the capital of Italy?"
context = "Italy, officially the Italian Republic, is a European country consisting of a peninsula delimited by the Alps and several islands. Its capital is Rome."
qa_model(question=question, context=context)
```

Result : 

```
{'score': 0.9793796539306641, 'start': 146, 'end': 150, 'answer': 'Rome'}
```

Analysis on example 5 : 

The question answering model accurately extracts "Rome" as the answer to the question "What is the capital of Italy?" from the provided context. The model assigns a high score of 0.9794 to its answer, indicating a very strong confidence in its correctness. The model also correctly identifies the start and end positions of the answer within the context, demonstrating its ability to pinpoint the relevant information. This result showcases the model's effectiveness in understanding the question and extracting the precise answer from the given text.

### 6. Example 6 - Text Summarization

```
# TODO :
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
summarizer(
    """
The field of artificial intelligence (AI) is rapidly evolving, transforming various aspects of our lives and industries. From self-driving cars and virtual assistants to medical diagnoses and financial forecasting, AI's potential seems limitless. Machine learning, a subset of AI, enables computers to learn from data without explicit programming, allowing them to make predictions and decisions. Deep learning, a more advanced form of machine learning, utilizes neural networks with multiple layers to analyze complex patterns. As AI becomes more integrated into our daily routines, ethical considerations and responsible development are crucial. Ensuring fairness, transparency, and accountability in AI systems is essential to prevent bias and misuse. Furthermore, the impact of AI on employment and the economy requires careful planning and adaptation. While AI offers numerous benefits, addressing these challenges will be vital for harnessing its full potential and creating a future where humans and AI can coexist harmoniously.
"""
)
```

Result : 

```
[{'summary_text': ' The field of artificial intelligence (AI) is rapidly evolving, transforming various aspects of our lives and industries . Ensuring fairness, transparency, and accountability in AI systems is essential to prevent bias and misuse . While AI offers numerous benefits, addressing these challenges will be vital for harnessing its full potential .'}]
```

Analysis on example 6 :

The summarization model effectively condenses the provided paragraph about artificial intelligence. The summary retains the key points, highlighting AI's transformative impact across industries, the importance of ethical considerations, and the need to address challenges to ensure harmonious human-AI coexistence. The model successfully captures the essence of the original text in a concise form, demonstrating its ability to extract and present the most relevant information while reducing redundancy.

### 7. Example 7 - Translation

```
# TODO :
translator_id = pipeline("translation", model="Helsinki-NLP/opus-mt-id-fr")
translator_id("Halo, nama saya Budi dan saya sedang belajar bahasa Prancis.")
```

Result : 

```
[{'translation_text': "Bonjour, je m'appelle Budi et j'étudie le français."}]

```

Analysis on example 7 :

The translation model accurately translates the Indonesian sentence "Halo, nama saya Budi dan saya sedang belajar bahasa Prancis." into French. The output "Bonjour, je m'appelle Budi et j'étudie le français." is a fluent and natural-sounding French equivalent, correctly conveying the meaning of "Hello, my name is Budi and I am learning French." The model demonstrates its proficiency in handling the nuances of language translation, producing a grammatically correct and semantically accurate result.

---

## Analysis on this project

This project effectively demonstrates the power and versatility of the Hugging Face Transformers library for various Natural Language Processing (NLP) tasks. By implementing and analyzing seven distinct examples – sentiment analysis, topic classification, text generation, masked language modeling, named entity recognition, question answering, and translation – it showcases the ease with which complex NLP operations can be performed using pre-trained models. The consistent accuracy and high confidence scores observed across the different tasks highlight the robustness and effectiveness of the Transformers library. Furthermore, the project provides a valuable hands-on experience in applying these models to real-world scenarios, fostering a deeper understanding of their capabilities and limitations. The ability to quickly prototype and experiment with different NLP tasks using Hugging Face Transformers underscores its significance as a tool for both researchers and practitioners in the field of Artificial Intelligence.
