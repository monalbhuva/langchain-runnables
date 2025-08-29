from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence , RunnableParallel ,RunnablePassthrough, RunnableLambda,RunnableBranch

load_dotenv()


model = ChatGroq(model="openai/gpt-oss-120b")
             
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='Write a detail report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following text \n {topic}',
    input_variables=['topic']
)

report_gen_chain = RunnableSequence(prompt1,model,parser)

branch_chain = RunnableBranch(
    (lambda x : len(x.split()) > 500, RunnableSequence(prompt2,model,parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain,branch_chain)

print(final_chain.invoke({'topic' : 'Russia vs Ukraine'}))