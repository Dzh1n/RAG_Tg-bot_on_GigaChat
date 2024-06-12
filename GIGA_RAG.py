import string, logging, nest_asyncio, asyncio, traceback
from   langchain_community.chat_models.gigachat import GigaChat
from   langchain_community.retrievers           import BM25Retriever
from   langchain.retrievers.ensemble            import EnsembleRetriever
from   langchain_community.document_loaders     import PyPDFLoader
from   langchain.text_splitter                  import RecursiveCharacterTextSplitter
from   chromadb.config                          import Settings
from   langchain_community.vectorstores         import Chroma
from   langchain_community.embeddings           import GigaChatEmbeddings
from   langchain.chains                         import RetrievalQA
from   langchain.chains.combine_documents       import create_stuff_documents_chain
from   langchain_core.prompts                   import ChatPromptTemplate
from   config                                   import WAY_FILE, TOKEN, PROMPT_TEXT, TOKEN_TG, USERS_ID
from   aiogram                                  import Bot, Dispatcher, executor, types
from   aiogram.contrib.fsm_storage.memory       import MemoryStorage
from   aiogram.contrib.middlewares.logging      import LoggingMiddleware
from   threading                                import Thread
nest_asyncio.apply()

documents = PyPDFLoader(WAY_FILE).load()
documents = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=200,
).split_documents(documents)

embeddings = GigaChatEmbeddings(credentials=TOKEN)

db = Chroma.from_documents(
    documents,
    embeddings,
    client_settings=Settings(anonymized_telemetry=False),)

llm = GigaChat(credentials=TOKEN)
document_chain = create_stuff_documents_chain(llm=llm, prompt=ChatPromptTemplate.from_template(PROMPT_TEXT))

bm25_retriever = BM25Retriever.from_documents(
    documents=documents,
    preprocess_func=lambda s: s.lower().translate(str.maketrans('', '', string.punctuation)).split(" "),
    k=3,)

ensemble_retriever = EnsembleRetriever(
    retrievers=[db.as_retriever(search_kwargs={"k": 2}), bm25_retriever],
    weights=[0.4, 0.6],)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=ensemble_retriever,
    return_source_documents=True,)

bot = Bot(token=TOKEN_TG)
dp  = Dispatcher(bot, storage=MemoryStorage())
dp.middleware.setup(LoggingMiddleware())

logging.basicConfig(level=logging.INFO)

try:
    async def ivoke_ai(bot, id_message, user_id, question):
        try:
            result = qa_chain.invoke({"query": question})
            try:
                await bot.edit_message_text(result['result'], chat_id=user_id, message_id=id_message)
            except:
                await bot.send_message(user_id, "Не удалось отправить ответ(")

        except Exception:
            print(traceback.format_exc(), '\nDATA: {0}\n\n'.format(id_message, question), 'ivoke_ai\n\n')

    def ivoke_ai_start(id_message, user_id, question):
        try:
            bot = Bot(token=TOKEN_TG)

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            loop.run_until_complete(ivoke_ai(bot, id_message, user_id, question))
            loop.run_until_complete(bot.close())

        except Exception:
            print(traceback.format_exc(), '\nDATA: {0}\n\n'.format(id_message, question), 'ivoke_ai_start\n\n')

    @dp.message_handler(commands="start")
    async def start(message: types.Message):
        try:
            if message.from_user.id in USERS_ID:
                await message.answer('Отправьте запрос для работы с GigaCHAT:')
            else:
                await message.answer('Отказано в доступе')

        except Exception:
            print(traceback.format_exc(), '\nDATA: {0}\n\n'.format(message), 'start\n\n')

    @dp.message_handler(lambda message: message.from_user.id in USERS_ID)
    async def send_prompt(message: types.Message):
        try:
            msg = await message.answer("Загрузка ответа...")
            Thread(target=ivoke_ai_start, args=(msg.message_id, message.from_user.id, message.text,), daemon=True).start()

        except Exception:
            print(traceback.format_exc(), '\nDATA: {0}\n\n'.format(message), 'send_prompt\n\n')

except Exception:
    print(traceback.format_exc(), "ALL")

if __name__ == "__main__":
    dp.register_message_handler(start, commands="start")
    print("START")
    executor.start_polling(dp, skip_updates=True)