def easy_choice_question(name):
    user_prompt = f"""
It is known that the creature in this picture is a {name} in disguise. Please generate five multiple-choice questions about the type of creature in the picture, with four options for each question, follow each question with an answer. 
Your answer should follow the following format:

<ID>. <question>? Answer: <answer>
    A. XXX
    B. XXX
    C. XXX
    D. XXX
    
Here are some examples:

1. Is there an animal in this picture that is in disguise? If so, what is it? Answer: A
   A. No
   B. Cat
   C. Dog
   D. Fox

2. What is the animal in this picture that is in disguise? Answer: C
   A. Whale
   B. Seahorse
   C. Clownfish
   D. Octopus

3. The creature in this picture is known for its excellent camouflage, what is its name? Answer: B
   A. Bee
   B. Cheetah
   C. Vulture
   D. Antelope

4. A camouflaged creature is hiding in this forest in the picture, who is it? Answer: D
   A. Wolf
   B. Tiger
   C. Rabbit
   D. Chameleon

*IMPORTANT*
1. The question stem should not contain any hint information about the answer.
2. The four candidate answers should all be names of living creatures and should be easily distinguishable to reduce the difficulty of the question.
3. Your answer should follow the following format:
    <ID>. <question>? Answer: <answer>
        A. XXX
        B. XXX
        C. XXX
        D. XXX
    """
    system_prompt = ""
    return system_prompt, user_prompt


def hard_choice_question(name):
    user_prompt = f"""
It is known that the creature in this picture is a {name} in disguise. Please generate five multiple-choice questions about the type of creature in the picture, with four options for each question, follow each question with an answer. 
Your answer should follow the following format:

<ID>. <question>? Answer: <answer>
    A. XXX
    B. XXX
    C. XXX
    D. XXX
    
Here are some examples:

What is the species of the creature in the picture? Answer: D
A. Gecko
B. Iguana
C. Salamander
D. Chameleon

Which type of creature is shown in the picture? Answer: C
A. Squid
B. Jellyfish
C. Octopus
D. Cuttlefish

What is the name of the animal depicted in the picture? Answer: B
A. Squid
B. Cuttlefish
C. Nautilus
D. Starfish

What species is represented in the image? Answer: A
A. Stick insect
B. Leaf insect
C. Praying mantis
D. Grasshopper

Which animal is featured in the picture? Answer: C
A. Mossy frog
B. Tree frog
C. Leaf-tailed gecko
D. Green anole

*IMPORTANT*
1. The question stem should not contain any hint information about the answer.
2. The four candidate answers should all be names of living creatures.
3. Increase the difficulty by selecting a more similar creature as a candidate answer when designing the options
4. Be careful not to mention "camouflage" and other clues indicating camouflage in the question, but directly ask the species of the organism to improve the difficulty of the question
5. Your answer should follow the following format:
    <ID>. <question>? Answer: <answer>
        A. XXX
        B. XXX
        C. XXX
        D. XXX
    """
    system_prompt = ""
    return system_prompt, user_prompt


def choice_question2llava_prompt(text):
    prompt = f"""
    You are an expert in camouflage biological monitoring. Your task is to detect camouflaged creatures in pictures and answer A multiple-choice question containing a text description and four choices. Please choose the answer you think is most appropriate from the four choices [A, B, C, D]. If you are not sure of the answer, please still choose the answer you think is most likely to be correct.
    *IMPORTANT*
    1.Your answer should be just one letter in [A, B, C, D]
    2.Don't interpret your answer in any way
    
    Question:
    {text}
    """

    return prompt
