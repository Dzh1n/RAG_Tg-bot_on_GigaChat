from langchain_community.chat_models.gigachat import GigaChat
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GigaChatEmbeddings
from langchain.chains import RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import string

# 3. Загрузите документ
loader = PyPDFLoader(r"****")
documents = loader.load()

# 4. Разделите документ на фрагменты
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=200,
)
documents = text_splitter.split_documents(documents)

# 5. Создайте векторную базу данных (Chroma)
embeddings = GigaChatEmbeddings(
    credentials="****")
db = Chroma.from_documents(
    documents,
    embeddings,
    client_settings=Settings(anonymized_telemetry=False),
)

# 6. Создайте цепочку QnA (RetrievalQA)
llm = GigaChat(
    credentials="****")

# 7. Используйте промпт из статьи
prompt = ChatPromptTemplate.from_template('''Ответь на вопрос пользователя. \\\n
Используй при этом только информацию из контекста. Если в контексте нет \\\n
информации для ответа, сообщи об этом пользователю.\n
Контекст: {context}\n
Вопрос: {input}\n
Ответ:'''
)

# 8. Создайте цепочку для комбинирования документов
document_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt
)

# 9. Создайте ретривер BM25
bm25_retriever = BM25Retriever.from_documents(
    documents=documents,
    preprocess_func=lambda s: s.lower().translate(str.maketrans('', '', string.punctuation)).split(" "),
    k=3,
)

# 10. Создайте ретривер с помощью эмбеддингов
embedding_retriever = db.as_retriever(search_kwargs={"k": 2})

# 11. Создайте ансамблевый ретривер
ensemble_retriever = EnsembleRetriever(
    retrievers=[embedding_retriever, bm25_retriever],
    weights=[0.4, 0.6],
)

# 12. Создайте цепочку RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=ensemble_retriever,
    return_source_documents=True,
)

# 13. Задайте вопрос и получите ответ
question = "Перечисли все Типы вредоносного ПО?"
result = qa_chain.invoke({"query": question})
print(result)

# 14. Запишите результат в файл
with open("output.txt", "w", encoding="utf-8") as file:
    file.write(f"Вопрос: {question}\n\n")
    file.write(f"Ответ: {result}\n\n")
    file.write(f"Использованный контекст:\n\n")
    for document in result["source_documents"]:
        file.write(f"{document.page_content}\n\n")