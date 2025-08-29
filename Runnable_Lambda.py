from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence , RunnableParallel , RunnablePassthrough, RunnableLambda

load_dotenv()

def word_count(text):
    return len(text.split())

model = ChatGroq(model="openai/gpt-oss-120b")
             
parser = StrOutputParser()

prompt = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

joke_gen_chain = RunnableSequence(prompt,model,parser)

prompt2 = PromptTemplate(
    template='Generate a Linkedin post about {topic}',
    input_variables=['topic']
)

parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'word_count' : RunnableLambda(word_count)
})

# parallel_chain = RunnableParallel({
#     'joke' : RunnablePassthrough(),
#     'word_count' : RunnableLambda(lambda x : len(x.split()))
# })

final_chain = RunnableSequence(joke_gen_chain,parallel_chain)

print(final_chain.invoke({'topic' : 'AI'}))