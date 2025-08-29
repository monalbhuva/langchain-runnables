from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence , RunnableParallel

load_dotenv()

model1 = ChatGroq(model="openai/gpt-oss-120b")

model2= ChatGroq(model="gemma2-9b-it")                

prompt1 = PromptTemplate(
    template='Generate A Tweet about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a Linkedin post about {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet' : RunnableSequence(prompt1,model1,parser),
    'linkedin' : RunnableSequence(prompt2,model2,parser)
})

result = parallel_chain.invoke({'topic' : 'AI'})
print(result)