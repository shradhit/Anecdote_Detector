# Anecdote Detector



Stories and anecdotes are often used by teachers, speakers, authors, and leaders to communicate abstract ideas and illustrate concepts. Stories by customers in business conversations (such as customer support) in social media channels are also important sources of information. Discovering these anecdotes from texts such as books, articles and blogs posts are currently done by human researchers. In this project, we aim to automatically identify these stories from such sources, so as to:
Improve the productivity of such researchers
Provide a story-detection component that can be used by other text analytics systems

To do this, we :
- Defined a ‘story’ is and its analytical components
- Used a Machine Learning based approach to identify potential regions of stories in unstructured text sources
- Build an application to do this.

The application that we build 
- automatically identifies regions of anecdotes (think of it as putting a <anecdote> </anecdote> tag around such regions of text)
- classifes them by certain labels (such as sports, films, business etc.) - these labels will be provided by us. 

Technology for this project:
- Machine learning: A field of computer science that gives computers the ability to learn without being explicitly programmed. Our proposed application will use ML to train the application to search for anecdotes from a given text.  We will use supervised learning techniques.
- Natural language processing (NLP): Computational techniques to automatically process unstructured human language. NLTK, Stanford’s CoreNLP, spaCy are some of the NLP libraries that will be used for developing the application. 

Phases for the project:
- Phase 1: Knowledge of ML and NLP, literature survey, build a dataset for use in supervised classification, identify ML features
- Phase 2: Build a supervised ML model; train and test iteratively
- Phase 3: Add a model for topic classification (classifying the anecdote into genres)
