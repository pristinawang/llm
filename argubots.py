"""This module contains argument bots. 
These agents should be able to handle a wide variety of topics and opponents.
They will be evaluated using methods in `eval.py`.
We've included a few to get your started."""
from openai import OpenAI
import logging
from rich.logging import RichHandler
from pathlib import Path
import random
import glob
from dialogue import Dialogue
from agents import Agent, ConstantAgent, LLMAgent
from kialo import Kialo
from tracking import default_client, default_model, default_eval_model
import numpy as np
import time
# Use the same logger as agents.py, since argubots are agents;
# we split this file 
# You can change the logging level there.
log = logging.getLogger("agents")    

#############################
## Define some basic argubots
#############################

# Airhead (aka Absentia or Acephalic) always says the same thing.

airhead = ConstantAgent("Airhead", "I know right???")

# Alice is a basic prompted LLM.  You are trying to improve on Alice.
# Don't change the prompt -- instead, make a new argubot with a new prompt.

alice = LLMAgent("Alice",
                 system="You are an intelligent bot who wants to broaden your user's mind. "
                        "Ask a conversation starter question.  Then, WHATEVER "
                        "position the user initially takes, push back on it. "
                        "Try to help the user see the other side of the issue. "
                        "Answer in 1-2 sentences. Be thoughtful and polite.")

############################################################
## Other argubot classes and instances -- add your own here! 
############################################################

class KialoAgent(Agent):
    """ KialoAgent subclasses the Agent class. It responds with a relevant claim from
    a Kialo database.  No LLM is used."""
    
    def __init__(self, name: str, kialo: Kialo):
        self.name = name
        self.kialo = kialo
                
    def response(self, d: Dialogue) -> str:

        if len(d) == 0:   
            # First turn.  Just start with a random claim from the Kialo database.
            claim = self.kialo.random_chain()[0]
        else:
            previous_turn = d[-1]['content']  # previous turn from user
            # Pick one of the top-3 most similar claims in the Kialo database,
            # restricting to the ones that list "con" arguments (counterarguments).
            neighbors = self.kialo.closest_claims(previous_turn, n=3, kind='has_cons')
            assert neighbors, "No claims to choose from; is Kialo data structure empty?"
            neighbor = random.choice(neighbors)
            log.info(f"[black on bright_green]Chose similar claim from Kialo:\n{neighbor}[/black on bright_green]")
            
            # Choose one of its "con" arguments as our response.
            claim = random.choice(self.kialo.cons[neighbor])
        
        return claim    
    
# Akiko doesn't use an LLM, but looks up an argument in a database.
  
akiko = KialoAgent("Akiko", Kialo(glob.glob("data/*.txt")))   # get the Kialo database from text files




class SomeAgent(Agent):
    """ KialoAgent subclasses the Agent class. It responds with a relevant claim from
    a Kialo database.  No LLM is used."""
    
    def __init__(self, name: str, kialo: Kialo, thresh: float, client: OpenAI = default_client):
        self.name = name
        self.kialo = kialo
        self.client=client
        self.thresh=thresh
        
                
    def response(self, d: Dialogue) -> str:

        if len(d) == 0:   
            # First turn.  Just start with a random claim from the Kialo database.
            claim = self.kialo.random_chain()[0]
        else:
            
            previous_turn = d[-1]['content']  # previous turn from user
            
            # Pick one of the top-3 most similar claims in the Kialo database,
            # restricting to the ones that list "con" arguments (counterarguments).
            score = self.kialo.get_maxscore(previous_turn, n=3, kind='has_cons')
            
            

            #print('score', score)
            if score > self.thresh:
                neighbors = self.kialo.closest_claims(previous_turn, n=3, kind='has_cons')
            
           
                assert neighbors, "No claims to choose from; is Kialo data structure empty?"
                neighbor = random.choice(neighbors)
                log.info(f"[black on bright_green]Chose similar claim from Kialo:\n{neighbor}[/black on bright_green]")
           
                # Choose one of its "con" arguments as our response.
                claim = random.choice(self.kialo.cons[neighbor])
            else:
                m=""
                for dia in d:
                    speaker=dia['speaker']
                    content=dia['content']
                    if content[-1].isalpha() or content[-1].isnumeric():
                        content=content+"."
                    dia=speaker+": "+content+"\n"
                    m=m+dia
                m=m.strip('\n')
                # print('-----------')
                # print(m.strip('\n'))
                # print('-------------')
                # name_turns="Generate a"+" "+ str(turns)+ " sentences dialogue between"+" "+a.name+" and "+b.name+". " 
                # a_persona_constyle= "The persona of "+a.name+": "+a.persona+". "+a.name+a.conversational_style.strip('You')+" "
                
                # starter='The dialogue should start with "('+a.name+") "+a.conversation_starters[0]+'"'
                # m=name_turns+a_persona_constyle+b_persona_constyle+starter

    
                response = self.client.chat.completions.create(messages=[
                                                                { "role": "user",        # input
                                                                "content": m }],
                                                    model="gpt-3.5-turbo-1106", temperature=0)
                #print('GPT:')

                response_gpt=response.choices[0].message.content
                while len(response_gpt)>(len(self.name)+1) and response_gpt[:len(self.name)+1]==self.name+":":
                    response_gpt=response_gpt.strip(self.name)   
                    response_gpt=response_gpt.strip(":")
                    response_gpt=response_gpt.strip(" ")
                # print('pre strip', response_gpt)
                # print(response_gpt[:len(self.name)])
                #response_gpt=response_gpt.strip(self.name+': ')
                # print(response_gpt)
                # print('++++++++++++++++++')
                # print('pre dialogue')
                # for k in d:
                #     print(k)
                # print('=============')
                claim=response_gpt

        
        return claim
class RAGAgent(Agent):
    """ KialoAgent subclasses the Agent class. It responds with a relevant claim from
    a Kialo database.  No LLM is used."""
    
    def __init__(self, name: str, kialo: Kialo, thresh: float, client: OpenAI = default_client):
        self.name = name
        self.kialo = kialo
        self.client=client
        self.thresh=thresh
        self.speaker=None
        
                
    def response(self, d: Dialogue) -> str:

        if len(d) == 0:   
            # First turn.  Just start with a random claim from the Kialo database.
            claim = self.kialo.random_chain()[0]
        else:
            
            previous_turn = d[-1]['content']  # previous turn from user
            
            # Pick one of the top-3 most similar claims in the Kialo database,
            # restricting to the ones that list "con" arguments (counterarguments).
            score = self.kialo.get_maxscore(previous_turn, kind='has_cons')
            
            

            #print('score', score)
            # if score > self.thresh:
            #     neighbors = self.kialo.closest_claims(previous_turn, n=3, kind='has_cons')
            
           
            #     assert neighbors, "No claims to choose from; is Kialo data structure empty?"
            #     neighbor = random.choice(neighbors)
            #     log.info(f"[black on bright_green]Chose similar claim from Kialo:\n{neighbor}[/black on bright_green]")
           
            #     # Choose one of its "con" arguments as our response.
            #     claim = random.choice(self.kialo.cons[neighbor])
            if True:
                #print('##d',d)
                m=""
                for dia in d:
                    speaker=dia['speaker']
                    content=dia['content']
                    if speaker!=self.name: self.speaker=speaker
                    if content[-1].isalpha() or content[-1].isnumeric():
                        content=content+"."
                    dia=speaker+": "+content+"\n"
                    m=m+dia
                ##Understanding speaker's meaning better
                ##Don't hullucinate
                ##Speaker reply type: Question. Statement-Pro, Con
                ##Get better Kialo data
                m2="The following is a conversation between "+self.speaker+" and "+self.name+".\n"\
                    "Turn "+self.speaker+"'s last reply into a more explicit reply with more information in the context of the whole conversation.\n"+\
                    "Weigh more recent sentences heavier than earlier sentences. The output reply must be in first person and don't mention the other speaker's name, "+ self.name+", in the output.\n"\
                    "Don't imagine stuff that "+self.speaker+" did not imply under the context of the conversation.\n"\
                    "Don't weigh in sentences by DemoUser, these are examples that show you what we mean by extractnig a more explicit reply."\
                    
                explicit = self.client.chat.completions.create(messages=[
                                                                { "role": "user",        # input
                                                                "content": m2 },
                                                                { "role": "user",        # input
                                                                "content": "DemoUser: The vegan diet is not an option for some people." },
                                                                { "role": "assistant",   # output
                                                                "content": "A vegan diet is not well-suited for vulnerable individuals or people with lifestyles requiring specialised nutrition, who may be unable to remove animal products from their diet." },
                                                                { "role": "user",        # input
                                                                "content": "DemoUser: Biden does not keep his promises." },
                                                                { "role": "assistant",   # output
                                                                "content": "In the first several months of his presidency, Biden has backpedaled on many promises he made during his campaign." },
                                                                { "role": "user",        # input
                                                                "content": m2 }],
                                                                model="gpt-3.5-turbo-1106", temperature=0)
                
                explicit_claim=explicit.choices[0].message.content
                
                #new_con=m.rsplit(self.speaker,1)[0] + self.speaker + ": "+explicit_claim
                ## Reevaluate explicit claim


                #print('##Speaker means', explicit_claim)
                result_kialo = self.kialo.closest_claims(explicit_claim, kind='has_cons')[0]
                docu = f'One possibly related claim from the Kialo debate website:\n\t"{result_kialo}"'
                if self.kialo.pros[result_kialo]:
                    docu += '\n' + '\n\t* '.join(["Some arguments from other Kialo users in favor of that claim:"] + self.kialo.pros[result_kialo])
                if self.kialo.cons[result_kialo]:
                    docu += '\n' + '\n\t* '.join(["Some arguments from other Kialo users against that claim:"] + self.kialo.cons[result_kialo])
                #print('##Docu:', docu)
                z1="How similar are this paragraph, "+explicit_claim+", and the following document?\n\n"\
                    "Document:\n"+docu+\
                    "Rate the similarity on a scale from 1 to 10 and only output the number."
                sim= self.client.chat.completions.create(messages=[
                                                                { "role": "user",        # input
                                                                "content": z1 }],
                                                    model="gpt-3.5-turbo-1106", temperature=0)
                sim=sim.choices[0].message.content
                #print('##Similar:', sim)
                sim=str(sim)
                if sim.isnumeric():
                    sim=int(sim)
                else:sim=5

                
                new_con=m.rsplit(self.speaker,1)[0] + self.speaker + ": "+explicit_claim
                
                
                if sim > 5:
                    m3="The following is a conversation between "+self.speaker+" and "+self.name+".\n"+new_con+"\n\n"\
                    "Help "+self.name+" form a response using the following information:\n"+docu+"\n\n"\
                    "Read the information and understand it. Later when you form response with the above information, forget the information is from Kialo.\n"\
                    "Pretend the information from Kialo is something you already know and don't mention you got the information from Kialo.\n"\
                    "When forming the response, look at the previous conversation to figure out what the topic is.\n"\
                    "After figuring out the topic, if the above information is related to the conversation, rely on the above information to form your response.\n"\
                    "Give fact. Give both pros and cons. Provide evidence from different angle.\n"\
                    "Don't repeat what "+self.name+" previously said.\n"\
                    +self.name+"is an ethical and intelligent person. "+self.name+"'s reply should reflect their values.\n"\
                    "The response should be first person from "+self.name+"'s point of view."
                else:
                    m3="The following is a conversation between "+self.speaker+" and "+self.name+".\n"+new_con+"\n\n"\
                    "Help "+self.name+" form a response.\n"\
                    "When forming the response, look at the previous conversation to figure out what the topic is.\n"\
                    "Your response should stay on the topic.\n"\
                    +self.name+"is an ethical and intelligent person. "+self.name+"'s reply should reflect their values.\n"\
                    "The response should be first person from "+self.name+"'s point of view."

                response = self.client.chat.completions.create(messages=[
                                                                { "role": "user",        # input
                                                                "content": m3 }],
                                                    model="gpt-3.5-turbo-1106", temperature=0)
                

                response_gpt=response.choices[0].message.content
                
                while len(response_gpt)>(len(self.name)+1) and response_gpt[:len(self.name)+1]==self.name+":":
                    response_gpt=response_gpt.strip(self.name)   
                    response_gpt=response_gpt.strip(":")
                    response_gpt=response_gpt.strip(" ")
                #print('##Response:', response_gpt)
                claim=response_gpt

        
        return claim
aragorn = RAGAgent("Aragorn", Kialo(glob.glob("data/*.txt")),7)

class AwsomAgent(Agent):
    """ AwsomAgent."""
    
    def __init__(self, name: str, kialo: Kialo, thresh: float, client: OpenAI = default_client):
        self.name = name
        self.kialo = kialo
        self.client=client
        self.thresh=thresh
        self.speaker=None
        self.used_pros={}
        self.used_cons={}
        
                
    def response(self, d: Dialogue) -> str:

        if len(d) == 0:   
            # First turn.  Just start with a random claim from the Kialo database.
            claim = self.kialo.random_chain()[0]
            #reset self.used_statements to keep track of used arguments
            self.used_pros={}
            self.used_cons={}
        else:
            time.sleep(5)
            
            previous_turn = d[-1]['content']  # previous turn from user
            
            # Pick one of the top-3 most similar claims in the Kialo database,
            # restricting to the ones that list "con" arguments (counterarguments).
            score = self.kialo.get_maxscore(previous_turn, kind='has_cons')
            
            

            
            if True:
                #print('##d',d)
                m=""
                for dia in d:
                    speaker=dia['speaker']
                    content=dia['content']
                    if speaker!=self.name: self.speaker=speaker
                    if content[-1].isalpha() or content[-1].isnumeric():
                        content=content+"."
                    dia=speaker+": "+content+"\n"
                    m=m+dia
                ##Understanding speaker's meaning better
                ##Don't hullucinate
                ##Speaker reply type: Question. Statement-Pro, Con
                ##Get better Kialo data
                ## Few shot prompting 
                m2="The following is a conversation between "+self.speaker+" and "+self.name+".\n"\
                    "Turn "+self.speaker+"'s last reply into a more explicit reply with more information in the context of the whole conversation.\n"+\
                    "Weigh more recent sentences heavier than earlier sentences. The output reply must be in first person and don't mention the other speaker's name, "+ self.name+", in the output.\n"\
                    "Don't imagine stuff that "+self.speaker+" did not imply under the context of the conversation.\n"\
                    "Don't weigh in sentences by DemoUser, these are examples that show you what we mean by extractnig a more explicit reply."\
                    
                explicit = self.client.chat.completions.create(messages=[
                                                                { "role": "user",        # input
                                                                "content": m2 },
                                                                { "role": "user",        # input
                                                                "content": "DemoUser: The vegan diet is not an option for some people." },
                                                                { "role": "assistant",   # output
                                                                "content": "A vegan diet is not well-suited for vulnerable individuals or people with lifestyles requiring specialised nutrition, who may be unable to remove animal products from their diet." },
                                                                { "role": "user",        # input
                                                                "content": "DemoUser: Biden does not keep his promises." },
                                                                { "role": "assistant",   # output
                                                                "content": "In the first several months of his presidency, Biden has backpedaled on many promises he made during his campaign." },
                                                                { "role": "user",        # input
                                                                "content": m2 }],
                                                                model="gpt-3.5-turbo-1106", temperature=0)
                
                explicit_claim=explicit.choices[0].message.content
                #print('##Claim', explicit_claim)



                ## Classify Speaker's Stance
                z1="The following is a conversation between "+self.speaker+" and "+self.name+".\n"+m+"\n"\
                    "Is "+self.speaker+"'s reply a question or a statement?\n"+\
                    "If it is a question, why is "+self.speaker+" asking this question? What is the possible reasoning behind "+self.speaker+"'s question?\n"\
                    "If it is a statement, what is "+ self.speaker+"'s stance on this matter?"
                classi = self.client.chat.completions.create(messages=[
                                                                { "role": "user",        # input
                                                                "content": z1 }],
                                                    model="gpt-3.5-turbo-1106", temperature=0)
                classification=classi.choices[0].message.content
                #print('##Classification', classification)

                # Weight "classification" and "explicit_claim"
                z3="The following is a conversation between "+self.speaker+" and "+self.name+".\n"+m+"\n"\
                    "Here we have two paragraph. Which describes "+self.speaker+"'s last reply more accuately?\n"+\
                    "Paragraph 1:\n"\
                    +explicit_claim+"\n\n"\
                    "Paragraph 2:\n"\
                    +classification+"\n\n"\
                    "If paragraph 1 describes"+self.speaker+"'s last reply in the above conversation more accurately, output a score of 1.\n"\
                    "If paragraph 2 describes"+self.speaker+"'s last reply in the above conversation more accurately, output a score of 2.\n"\
                    "If both paragraphs describe"+self.speaker+"'s last reply in the above conversation equally accurately, output a score of 3.\n"\
                    "Just output a score. Nothing else.\n"
                weigh_classi_claim = self.client.chat.completions.create(messages=[
                                                                { "role": "user",        # input
                                                                "content": z3 }],
                                                    model="gpt-3.5-turbo-1106", temperature=0)
                weigh_classi_claim=weigh_classi_claim.choices[0].message.content
                weight=str(weigh_classi_claim)
                #print('##Weight', weigh_classi_claim)

                if weight=='1':
                    speaker_meaning=explicit_claim

                elif weight=='2':
                    speaker_meaning=classification
                elif weight =='3':
                    speaker_meaning=explicit_claim+'\n'+classification

                else: #This happens when gpt output something other than a score 1, 2, or 3. In this case, we back off to weight 1.
                    speaker_meaning=explicit_claim
                
                # Get relevent info from Kialo
                result_kialo = self.kialo.closest_claims(speaker_meaning, kind='has_cons')[0]
                docu = f'One possibly related claim from the Kialo debate website:\n\t"{result_kialo}"'

                ## Get unused pro statement -> store in List: pl
                if len(self.kialo.pros[result_kialo])>0:
                    p_i=0
                    pro=self.kialo.pros[result_kialo][p_i]
                    if pro in self.used_pros:
                        p_i+=1
                        while pro in self.used_pros and p_i < len(self.kialo.pros[result_kialo]):
                            
                            pro=self.kialo.pros[result_kialo][p_i]
                            p_i+=1
                        if pro in self.used_pros and p_i>=len(self.kialo.pros[result_kialo]):
                            #print('Claim', result_kialo)
                            #print('old pro', pro)
                            ask1="Give me an argument supporting this statement: "+result_kialo
                            pro_gpt = self.client.chat.completions.create(messages=[
                                                                { "role": "user",        # input
                                                                "content": ask1 }],
                                                    model="gpt-3.5-turbo-1106", temperature=0)
                

                            pro_gpt=pro_gpt.choices[0].message.content

                            pro=pro_gpt
                            #print('new pro', pro)
                        pl=[pro]
                        self.used_pros[pro]='pro'
                    else:
                        self.used_pros[pro]='pro'
                        pl=[pro]
                else:
                    pl=[]

                ## Get unused con statement -> store in List: cl
                if len(self.kialo.cons[result_kialo])>0:
                    c_i=0
                    con=self.kialo.cons[result_kialo][c_i]
                    if con in self.used_cons:
                        c_i+=1
                        while con in self.used_cons and c_i < len(self.kialo.cons[result_kialo]):
                            
                            con=self.kialo.cons[result_kialo][c_i]
                            c_i+=1
                        if con in self.used_cons and c_i>=len(self.kialo.cons[result_kialo]):
                            #print('Claim', result_kialo)
                            #print('old con', con)
                            ask1="Give me an argument opposing this statement: "+result_kialo
                            con_gpt = self.client.chat.completions.create(messages=[
                                                                { "role": "user",        # input
                                                                "content": ask1 }],
                                                    model="gpt-3.5-turbo-1106", temperature=0)
                

                            con_gpt=con_gpt.choices[0].message.content

                            con=con_gpt
                            #print('new con', con)
                        cl=[con]
                        self.used_cons[con]='con'
                    else:
                        self.used_cons[con]='con'
                        cl=[con]
                else:
                    cl=[]

                if pl:
                    docu += '\n' + '\n\t* '.join(["Some arguments from other Kialo users in favor of that claim:"] + pl)
                if cl:
                    docu += '\n' + '\n\t* '.join(["Some arguments from other Kialo users against that claim:"] + cl)

                # print('ANALYZE KIALO:')
                # print("THE CLAIM:")
                # print(result_kialo)
                # print("PROS")
                # print(self.kialo.pros[result_kialo])
                # print("CONS")
                # print(self.kialo.cons[result_kialo])
                # print('##Docu:', docu)

                # z2="How similar are this paragraph, "+speaker_meaning+", and the following document?\n\n"\
                #     "Document:\n"+docu+\
                #     "Rate the similarity on a scale from 1 to 10 and only output the number."
                # sim= self.client.chat.completions.create(messages=[
                #                                                 { "role": "user",        # input
                #                                                 "content": z2 }],
                #                                     model="gpt-3.5-turbo-1106", temperature=0)
                # sim=sim.choices[0].message.content
                # print('##Similar:', sim)
                # sim=str(sim)
                # if sim.isnumeric():
                #     sim=int(sim)
                # else:sim=5

                
                #new_con=m.rsplit(self.speaker,1)[0] + self.speaker + ": "+explicit_claim
                
                sim=10
                if sim > 5: 
                    m3="The following is a conversation between "+self.speaker+" and "+self.name+".\n"+m+"\n\n"\
                    "The last reply of "+self.speaker+" can be understood more clearly using the following explanation.\n"\
                    +"Explanation: "+speaker_meaning+"\n\n"\
                    "When forming a response, it is important to understand "+self.speaker+"'s meaning of their last sentence with the explanation above.\n"\
                    "Help "+self.name+" form a response using the following document:\n"+docu+"\n\n"\
                    "Read the document and understand it. Later when you form response with the above document, forget the information of the document is from Kialo.\n"\
                    "Pretend the document from Kialo is something you already know and don't mention you got the document from Kialo.\n"\
                    "When forming the response, look at the previous conversation to figure out what the topic is.\n"\
                    "After figuring out the topic, if the above document is related to the conversation, rely on the above information from the document to form your response.\n"\
                    "If the above explanation of the last reply of "+self.speaker+" is a statement, does this viewpoint align with "+self.name+"'s viewpoint?\n"\
                    "If the viewpoint align with "+self.name+"'s viewpoint, then invite "+self.speaker+" to think about some opposite opions that people opposing this viewpoint have. Use the document to give some opposite ideas.\n"
                    "If the viewpoint does not align with "+self.name+"'s viewpoint, ask "+self.speaker+" about their reasons and provide reasons on why "+self.name+" have this viewpoint\n"\
                    "If the above explanation of the last reply of "+self.speaker+" is a question, give "+self.name+"'s answer with thorough and intelligent reasons using the above document.\n"\
                    "Look at the conversation between "+self.speaker+" and "+self.name+" again and don't repeat what "+self.name+" previously said.\n"\
                    +self.name+"is an ethical and intelligent person. "+self.name+"'s reply should reflect their values.\n"\
                    "The response should be first person from "+self.name+"'s point of view."
                else:
                    m3="The following is a conversation between "+self.speaker+" and "+self.name+".\n"+m+"\n\n"\
                    "The last reply of "+self.speaker+" can be understood more clearly using the following explanation.\n"\
                    +"Explanation: "+speaker_meaning+"\n\n"\
                    "When forming a response, it is important to understand "+self.speaker+"'s meaning of their last sentence with the explanation above.\n"\
                    "Help "+self.name+" form a response.\n"\
                    "When forming the response, look at the previous conversation to figure out what the topic is.\n"\
                    "Your response should stay on the topic.\n"\
                    "If the above explanation of the last reply of "+self.speaker+" is a statement, does this viewpoint align with "+self.name+"'s viewpoint?\n"\
                    "If the viewpoint align with "+self.name+"'s viewpoint, then invite "+self.speaker+" to think about some opposite opions that people opposing this viewpoint have.\n"
                    "If the viewpoint does not align with "+self.name+"'s viewpoint, ask "+self.speaker+" about their reasons and provide reasons on why "+self.name+" have this viewpoint\n"\
                    "If the above explanation of the last reply of "+self.speaker+" is a question, give "+self.name+"'s answer with thorough and intelligent reasons.\n"\
                    "Look at the conversation between "+self.speaker+" and "+self.name+" again and don't repeat what "+self.name+" previously said.\n"\
                    +self.name+"is an ethical and intelligent person. "+self.name+"'s reply should reflect their values.\n"\
                    "The response should be first person from "+self.name+"'s point of view."

                response = self.client.chat.completions.create(messages=[
                                                                { "role": "user",        # input
                                                                "content": m3 }],
                                                    model="gpt-3.5-turbo-1106", temperature=0)
                

                response_gpt=response.choices[0].message.content
                #print('OG RESPONSE', response_gpt)
                while len(response_gpt)>(len(self.name)+1) and response_gpt[:len(self.name)+1]==self.name+":":
                    response_gpt=response_gpt.strip(self.name)   
                    response_gpt=response_gpt.strip(":")
                    response_gpt=response_gpt.strip(" ")
                #print('##Response:', response_gpt)
                response_gpt=response_gpt.split(self.name+':',1)[0]
                response_gpt=response_gpt.split(self.speaker+':',1)[0]
                claim=response_gpt

        
        return claim


awsom = AwsomAgent("Awsom", Kialo(glob.glob("data/*.txt")),7)

class AkikiAgent(Agent):
    """ KialoAgent subclasses the Agent class. It responds with a relevant claim from
    a Kialo database.  No LLM is used."""
    
    def __init__(self, name: str, kialo: Kialo, thresh: float, weight_const: int = 4,client: OpenAI = default_client):
        self.name = name
        self.kialo = kialo
        self.client=client
        self.thresh=thresh
        self.w_const=weight_const
        self.pre_index=None 
        
        
                
    def response(self, d: Dialogue) -> str:

        if len(d) == 0:   
            # First turn.  Just start with a random claim from the Kialo database.
            claim = self.kialo.random_chain()[0]
        else:
            
            previous_turn = d[-1]['content']  # previous turn from user
            
            # Pick one of the top-3 most similar claims in the Kialo database,
            # restricting to the ones that list "con" arguments (counterarguments).
            score = self.kialo.get_maxscore(previous_turn, kind='has_cons')
            
            

            #print('score', score)
            if score > self.thresh:
                neighbors = self.kialo.closest_claims(previous_turn, n=3, kind='has_cons')
            
           
                assert neighbors, "No claims to choose from; is Kialo data structure empty?"
                neighbor = random.choice(neighbors)
                log.info(f"[black on bright_green]Chose similar claim from Kialo:\n{neighbor}[/black on bright_green]")
           
                # Choose one of its "con" arguments as our response.
                claim = random.choice(self.kialo.cons[neighbor])
            else:
                nums=[]
                # for i in range(len(d)):
                #     nums.append(4^i)
                # print('nums',nums)
                
                w_i=0
                score_vs=[]
                for i in range(len(d)):
                    if d[i]['speaker']!=self.name:
                        nums.append(self.w_const**(w_i))
                        w_i+=1
                        #print(d[i])
                        this_turn = d[i]['content']  # previous turn from user
           
                        score = self.kialo.get_scores(this_turn, kind='has_cons')
                        score_vs.append(score)
                weights = [x / sum(nums) for x in nums]
                #print(nums)
                #print(weights)

                final_scores=np.zeros(len(score_vs[0]))
              
                for i in range(len(score_vs)):
                    w=weights[i]
                    s=score_vs[i]
                    #print('s',s)
                    #print('wxs',w*s)
                    
                    final_scores=final_scores+np.array(w*s)
                    
                #print('final scores',final_scores)   ##Stopped here

                index=np.argmax(final_scores)
                if index==self.pre_index:
                    ind=np.argpartition(final_scores,-3)[-3:]
                    index=np.random.choice(ind)
                self.pre_index=index

                #print('index', index,final_scores[index])
                
                
                claim=self.kialo.get_closest_byindex(this_turn, index=index, kind='has_cons')

        
        return claim
akiki = AkikiAgent("Akiki", Kialo(glob.glob("data/*.txt")),4,5) #og2
###########################################
# Define your own additional argubots here!
###########################################
