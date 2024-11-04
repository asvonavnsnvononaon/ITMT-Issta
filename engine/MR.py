import torch
import numpy
import json
import torch
# Load the workbook
import transformers
import torch
from altair.vegalite.v5.theme import theme
from huggingface_hub import login
import os
from tqdm import tqdm
import re
import glob

def Question_3(MR, traffic_rule,sort, road_network, objects_environment, maneuver):
    prompt = [
        {"role": "system", "content": '''# CONTEXT # Based on the list of close-ended yes or no questions, generate a JSON answer.
        #Key Concepts# 1. traffic rule: Define how the ego-vehicle should maneuver in the specific driving scenario. The ontology elements in driving scenario are classified into road_network and object_environment.
                            2. maneuver: A specific action or movement that an ego-vehicle performs.
                            3. Road Network: Road elements are specified in the traffic rule, such as lanes, lines and crosswalks.
                            4. Object/Environment: One object or environment is specified in the traffic rule.
    # OBJECTIVE # Provide yes/no answers to the given questions based on the provided MR and traffic rule.
    The elements of MR are:MR template.
Scenario: Ego-vehicle detects the <category>
Given the Road Network
When ITMI <modify> the Object/Environment
Then ego-vehicle should <maneuver>
    # STYLE # Generate a JSON object with answers for all questions in the following format:
    IMPORTANT: 
    - Only return the JSON object, nothing else.
    - The 'answers' key must contain a list of strings, either 'yes' or 'no'.
    - The number of answers must exactly match the number of questions.
    - Answer 'no' if there's not enough information to answer confidently.'''},
        {"role": "user", "content": f'''MR: Scenario: Ego-vehicle detects the traffic sign
    Given the anyroad
    When ITMI add STOP sign on the roadside
    Then ego-vehicle should stop
    Traffic rule: The stop sign means come to a complete stop, yield to pedestrians or other vehicles, and then proceed carefully. 
    Questions:
    1. Is the traffic rule supported by MR?
    2. Are all parts of the MR consistent with each other?
    3. Is the chosen category appropriate for the described Object/Environment?
    4. Does the traffic rule support the specified logic "In the Road Network, Object/Environment cause the ego-vehicle to maneuver"?
    5. Are Road Network, Object/Environment, and all mentioned in the traffic rule?'''},
        {"role": "assistant", "content": '{"answers": ["yes", "yes", "yes", "yes", "no"]}'},
        {"role": "user", "content": f'''MR: {MR}
    Traffic rule: {traffic_rule}

    Questions:
    1. Is the traffic rule supported by MR?
    2. Are all parts of the MR consistent with each other?
    3. Is the chosen category appropriate for the described Object/Environment?
    4. Does the traffic rule support the specified logic "In the Road Network, Object/Environment cause the ego-vehicle to "?
    5. Are Road Network, Object/Environment, and all mentioned in the traffic rule?'''}
    ]
    return prompt
def parse_json_with_retry(json_str):
    try:
        parsed = json.loads(json_str)
        if parsed == []:
            return None
        return parsed
    except json.JSONDecodeError:
        return None
def calculate_score(answers):
    score = 0
    total_answers = 0
    for question, answer_list in answers.items():
        total_answers += len(answer_list)
        for answer in answer_list:
            if answer.lower() == 'no':
                score += 1
    return score / total_answers if total_answers > 0 else 0
def Question_4(MR, traffic_rule,sort, road_network, objects_environment):
    prompt = [
        {"role": "system", "content": '''Generate a sample diffusion inpainting prompt based on the given traffic rule and scenario. 
        Provide ONLY the prompt, with no additional explanation or content.
        This prompt should describe the scene from the camera's perspective, focusing on the traffic rule.
        Ensure the prompt is faithful to the original text and captures the key visual elements.
        Describe the scene from the camera's perspective mounted on the ego vehicle that must change its state (e.g., yield, stop). Focus on the visual elements of the road network and environment without explicitly mentioning the ego vehicle as a subject.
        IMPORTANT: Limit the prompt to a maximum of 50 words.
        Dot not show any independent vehicle contorl wards like slow down, yield,  prepare to stop in this prompt!.'''},
        {"role": "user",
         "content": """Bicycle riders are vulnerable users and do not have the same protections as people in vehicles and can be seriously injured or killed in a crash. As a driver, it's your responsibility to help keep bicycle riders safe.
         road_network: any road,
        objects_environment: a bicycle rider"""},
        {"role": "assistant", "content": """a bicycle rider on the road"""},
        {"role": "user",
         "content": """Do not put your vehicle in the path of a large, heavy vehicle when it's turning – you may be crushed. It's safest to stay behind and wait until the vehicle has completed the turn. There are also rules that must be obeyed.
             road_network: any intersection,
            objects_environment: heavy vehicle when it's turning"""},
        {"role": "assistant",
         "content": """A heavy vehicle is turning in the intersection"""},
        {"role": "user",
         "content": """Do not put your vehicle in the path of a large, heavy vehicle when it's turning – you may be crushed. It's safest to stay behind and wait until the vehicle has completed the turn. There are also rules that must be obeyed.
                 road_network: any intersection,
                objects_environment: heavy vehicle when it's turning"""},
        {"role": "assistant",
         "content": """A heavy vehicle is turning in the intersection"""},
        {"role": "user", "content": f'''Traffic rule: {traffic_rule},road_network:{road_network},objects_environment:{objects_environment}'''},
    ]
    return prompt
class MRGenerator:
    def __init__(self):
        model_id = "arcee-ai/Llama-3.1-SuperNova-Lite"#"meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",)
        self.system_prompt = {"role": "system",
                              "content": '''# CONTEXT #You are an expert in traffic rules and scene analysis.
                            #Key Concepts# 1. traffic rule: Define how the ego-vehicle should maneuver in the specific driving scenario. The ontology elements in driving scenario are classified into road_network and object_environment.
                            2. maneuver: A specific action or movement that an ego-vehicle performs.
                            3. road_network: Road elements are specified in the traffic rule, such as lanes, lines and crosswalks.
                            4. object_environment: One object or environment is specified in the traffic rule.'''}

        self.terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        self.max_new_tokens=1024
        self.do_sample = True
        self.temperature = 0.1
        self.top_p = 0.9
        self.data_list = []
        self.L =5

    def apply_MR(self, categorys, road_network, object_environments, maneuver):
        MRs = []
        for i in range(len(object_environments)):
            category = categorys[i]
            road_network = road_network
            maneuver = maneuver
            object_environment = object_environments[i]
            if category == "weather":
                MR = f"""Scenario: Ego-vehicle detects the {category}\nWhen ITMI repalce environment into {object_environment}\nThen ego-vehicle should {maneuver}"""
            else:
                MR = f"""Scenario: Ego-vehicle detects the {category}\n Given the {road_network}\nWhen ITMI add {object_environment}\nThen ego-vehicle should {maneuver}"""
            MRs.append(MR)
        return MRs


    def find_maneuver(self, prompt):
        while(1):
            user_message = {
                "role": "user",
                "content": f"""
                # OBJECTIVE #  Your task is to find the ego-vehicle's most dangerous maneuver in the traffic rules.
                # STYLE # The supported maneuvers are limited to the following types:
                slow down, stop, turn, turn left, turn right, keep the same.
                # NOTICE # Some maneuvers could lead to the listed types, e.g., 'yield' can be interpreted as 'slow down'.
                Generate answer for maneuver only. Do not generate other output.
                # EXAMPLE # 
                Example Text: If you are driving on an unpaved road that intersects with a paved road, you must yield the right-of-way to vehicles traveling on the paved road.
                Example Answer: slow down
                Example Text: Steady Red Light (Stop) Stop before entering the crosswalk or intersection. You may turn right unless prohibited by law. You may also turn left if both streets are one way, unless prohibited by law. You must yield to all pedestrians and other traffic lawfully using the intersection.
                Example Answer: stop
                ===== END OF EXAMPLE ======
                Text: {prompt}
                Answer: """
            }
            self.conversation_history.append(user_message)

            answer = self.LLM(self.conversation_history)
            if answer in ["slow down", "stop", "turn", "turn left", "turn right", "keep the same"]:
                self.add_assistant_response(answer)
                return answer
    def find_road_network(self, prompt):
        user_message = {
            "role": "user",
            "content": f"""
            # OBJECTIVE #  Your task is to find the most dangerous road_network where ego-vehicle will violate the traffic rule.
            # NOTICE # Generate answer for road_network only. Do not generate other output. Before answering, verify the road_network describes road, not traffic participants.
            road_network only refers to road types, such as: intersection, one-way street, two-way street, roundabout, highway, freeway, residential street, rural road, urban street, bridge, tunnel, parking lot, alley, T-junction, divided highway, bike lane, etc. 
            road_network should not conflict with object_environments. For instance, avoid conflicts between a red arrow light and the lane where a red cross-shaped light or arrow light is active. In case of any conflicts, set road_network to "any road". The answer should not exceed three words.
            # EXAMPLE # 
            Example Text: If you are driving on an unpaved road that intersects with a paved road, you must yield the right-of-way to vehicles traveling on the paved road.
            Example Answer: the unpaved road 
            Example Text: Steady Red Light (Stop) Stop before entering the crosswalk or intersection. You may turn right unless prohibited by law. You may also turn left if both streets are one way, unless prohibited by law. You must yield to all pedestrians and other traffic lawfully using the intersection.
            Example Answer: the crosswalk
            ===== END OF EXAMPLE ======
            Text: {prompt}
            Answer: """
        }
        self.conversation_history.append(user_message)
        answer = self.LLM(self.conversation_history)
        self.add_assistant_response(answer)
        return answer
    def check_conflict(self,prompt):
        user_message = {
            "role": "user",
            "content": f"""
            # OBJECTIVE # Please check if the road_network describes traffic participants rather than the road itself. If it describes traffic participants, return True; otherwise, return False.
            road_network: {prompt}
            Answer: """
        }
        temp_conversation_history = self.conversation_history
        temp_conversation_history.append(user_message)
        answer = self.LLM_1(temp_conversation_history)
        if answer =="True":
            answer = "any road"
        else:
            answer = prompt
        return answer
    def find_category(self,maneuver,object_environments):
        answers = []
        if maneuver=="keep the same":
            for i in range(len(object_environments)):
                answer = "equality MRs"
                answers.append(answer)
        else:
            for i in range(len(object_environments)):
                Q=object_environments[i]
                messages = [
                    {"role": "system", "content": '''# CONTEXT # You are an expert in traffic rules and scene analysis.
                    # OBJECTIVE # Your task is to classify the input into one of these elements:
                    traffic sign, traffic signal, road marking, traffic participant, other road infrastructure, weather.
                    The output must be one of these elements, no other output!'''},
                    {"role": "user",
                     "content": """the right-of-way to vehicles traveling on the paved road"""},
                    {"role": "assistant", "content": """"traffic participant"""},
                    {"role": "user", "content": """STOP sign"""},
                    {"role": "assistant", "content": """traffic sign"""},
                    {"role": "user", "content": """a school bus with flashing red lights"""},
                    {"role": "assistant", "content": """traffic participant"""},
                    {"role": "user", "content": f" {Q}"}, ]
                outputs = self.pipeline(
                    messages,
                    max_new_tokens=self.max_new_tokens,
                    eos_token_id=self.terminators,
                    do_sample=self.do_sample,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    pad_token_id=self.pipeline.tokenizer.eos_token_id
                )
                answer = outputs[0]["generated_text"][-1]['content']
                valid_answers = [
                    "traffic sign",
                    "traffic signal",
                    "road marking",
                    "traffic participant",
                    "other road infrastructure",
                    "weather"
                ]

                if answer not in valid_answers:
                    answer = "other road infrastructure"


                answers.append(answer)
        return answers
    def find_object_environment(self,prompt):
        user_message = {
            "role": "user",
            "content": f"""
                    # OBJECTIVE #  Your task is to find all suitable object_environment, which will lead ego-vehicle to take the most dangerous maneuver.
                    # NOTICE # Generate answer for object_environment only. Do not generate other output.
                    Each item in the answer must be a single object_environment. Sentences containing "or" should be split into separate content.
                    Before answering, verify the logic: "In the road_network, object_environment cause the ego-vehicle to maneuver". If this logic doesn't hold true, please don't output this object_environment.
                    # EXAMPLE # 
                    Example Text: If you are driving on an unpaved road that intersects with a paved road, you must yield the right-of-way to vehicles traveling on the paved road.
                    Example Answer: "the right-of-way to vehicles traveling on the paved road"
                    Example Text: Steady Red Light (Stop) Stop before entering the crosswalk or intersection. You may turn right unless prohibited by law. You may also turn left if both streets are one way, unless prohibited by law. You must yield to all pedestrians and other traffic lawfully using the intersection.
                    Example Answer: "a steady red light"
                    Example Text: If an emergency medical vehicle, law enforcement vehicle, fire truck, tow truck, utility service vehicle, Texas Department of Transportation vehicle (TxDOT, or other highway construction or maintenance vehicle) is stopped on the road with its lights activated (the lights are on or flashing), then the driver is required: 1. To reduce his/her speed to 20 mph below the speed limit; or 2. Move out of the lane closest to the emergency medical vehicle, law enforcement vehicle, fire truck, tow truck or a TxDOT vehicle if the road has multiple lanes traveling in the same direction.
There are other instances where it is important to be observant of vehicles stopped on the road. Mail, delivery, and trash-collection vehicles often make frequent stops in the roadway. Drivers must proceed with caution, and, if possible, change lanes before safely passing one of these vehicles on the road.
                    Example Answer: "a stopped emergency medical vehicle with activated lights", "a stopped law enforcement vehicle with activated lights", "the stopped fire truck with activated lights", "the stopped tow truck with activated lights", "the stopped utility service vehicle with activated lights", "the stopped Texas Department of Transportation vehicle with activated lights", "a stopped highway construction vehicle with activated lights", "a stopped highway maintenance vehicle with activated lights", "a stopped mail vehicle, the stopped delivery vehicle", "a stopped trash-collection vehicle"
                    Example Text: Lane signal lights indicate: (1) When the green arrow light is on, allow vehicles in the lane to pass in the direction indicated; (2) When the red cross-shaped light or arrow light is on, vehicles in the lane are prohibited from passing.
                    Example Answer: "a red cross-shaped light","a red arrow light"
                    Example Text: Flashing red light: Vehicles and streetcars/trams must stop at the stopping point before proceeding.
                    Example Answer: "flashing red light"
                    Example Text: Slow down on wet road. Do not suddenly turn, speed up, or stop.
                    Example Answer: "Wet"
                    ===== END OF EXAMPLE ======
                    Text: {prompt}
                    Answer: """
        }
        self.conversation_history.append(user_message)
        answer = self.LLM(self.conversation_history)
        self.add_assistant_response(answer)
        answer = answer.strip('"')
        answer = [item.strip().strip('"').strip("'") for item in answer.split(',')]
        return answer
    def LLM(self,messages):
        outputs = self.pipeline(
            messages,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.terminators,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_p=self.top_p,
            pad_token_id=self.pipeline.tokenizer.eos_token_id
        )
        answer = outputs[0]["generated_text"][-1]['content']

        return answer
    def LLM_1(self,messages):
        outputs = self.pipeline(
            messages,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.terminators,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_p=self.top_p,
            pad_token_id=self.pipeline.tokenizer.eos_token_id
        )
        answer = outputs[0]["generated_text"][-1]['content']
        return answer
    def calculate_score_1(self,MR, traffic_rule, category, road_network, object_environment, maneuver):
            messages = Question_3(MR, traffic_rule, category, road_network, object_environment, maneuver)
            outputs = self.pipeline(
            messages,
            max_new_tokens=512,
            eos_token_id=self.terminators,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_p=self.top_p,
            pad_token_id=self.pipeline.tokenizer.eos_token_id
        )
            answer = outputs[0]["generated_text"][-1]['content']
            answer = parse_json_with_retry(answer)
            score = calculate_score(answer)
            return score
    def find_prompt(self,MRs, traffic_rule, categorys, road_network, object_environments, maneuver):
        diffusion_prompts = []
        for i in range(len(object_environments)):
            category = categorys[i]
            object_environment = object_environments[i]
            MR = MRs[i]
            if category == "weather":
                diffusion_prompt = object_environment
            else:
                messages = Question_4(MR, traffic_rule,category,road_network , object_environment)
                outputs = self.pipeline(
                    messages,
                    max_new_tokens=512,
                    eos_token_id=self.terminators,
                    do_sample=self.do_sample,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    pad_token_id=self.pipeline.tokenizer.eos_token_id
                )
                answer = outputs[0]["generated_text"][-1]['content']
                diffusion_prompt = answer
            diffusion_prompts.append(diffusion_prompt)

        return diffusion_prompts

    def find_elements(self,traffic_rule):
        self.conversation_history = [self.system_prompt]
        prompt = traffic_rule
        maneuver = self.find_maneuver(prompt)
        road_network = self.find_road_network(prompt)
        object_environments = self.find_object_environment(prompt)
        road_network = self.check_conflict(road_network)
        categorys = self.find_category(maneuver, object_environments)
        MRs = self.apply_MR(categorys, road_network, object_environments, maneuver)
        return prompt,maneuver,road_network,object_environments,categorys,MRs

    def __call__(self, traffic_rule):
        compare_score = 100
        for j in range(self.L):
            llm_scores = 0  # 用于存储每个LLM的总分
            llm_count = 0  # 用于存储每个LLM的有效评分次数


            prompt, maneuver, road_network, object_environments, categorys, MRs = self.find_elements(traffic_rule)
            for i in range(len(object_environments)):
                category = categorys[i]
                object_environment = object_environments[i]
                MR = MRs[i]
                score = self.calculate_score_1(MR, prompt, category, road_network, object_environment, maneuver)
                llm_scores += score
                llm_count += 1
            llm_scores = llm_scores/llm_count
            if compare_score > llm_scores:
                compare_score = llm_scores
                MRs_ = MRs
                categorys_ = categorys
                object_environments_ = object_environments
                road_network_ = road_network
                maneuver_ = maneuver
        diffusion_prompts = self.find_prompt(MRs_, prompt, categorys_, road_network_, object_environments_, maneuver_)
        print(maneuver_)
        return MRs_, prompt, categorys_, road_network_, object_environments_, maneuver_,diffusion_prompts,compare_score

    def add_assistant_response(self, response):
        assistant_message = {
            "role": "assistant",
            "content": response
        }
        self.conversation_history.append(assistant_message)

    def get_conversation_history(self):
        return self.conversation_history

import random

if __name__ == "__main__":


    Texas = [
        'If you are driving on an unpaved road that intersects with a paved road, you must yield the right-of-way to vehicles traveling on the paved road.',
        'When approaching a railroad grade crossing, stop between 15 and 50 feet from the nearest rail if: 1. A clearly visible railroad signal warns of an approaching train 2. A crossing gate is lowered or a flag person warns of an approaching train 3. A driver is required to stop by an official traffic-control device or a traffic-control signal 4. An approaching train is within about 1,500 feet of the crossing. The train will produce an audible signal to identify the immediate hazard. 6. An approaching train is visible and in close proximity to the crossing',
        'A Flashing Red Light Stop completely before entering the crosswalk or intersection, then proceed when you can do so safely. Vehicles on the intersecting road may not have to stop',
        'Steady Yellow Light (Caution) A steady yellow light warns drivers to use caution and to alert them the light is about to change to red. You must STOP before entering the nearest crosswalk at the intersection if you can do so safely. If a stop cannot be made safely, then you may proceed cautiously through the intersection before the light changes to red.',
        'You must yield the right-of-way to police cars, fire trucks, ambulances, and other emergency vehicles sounding a siren, bell, or flashing red light. If traffic allows, pull to the right edge of the road and stop. If you are unable to pull over to the right, slow down and leave a clear path for the emergency vehicle.',
        'Slow down on wet road. Do not suddenly turn, speed up, or stop.',
        'Road ahead makes a gradual curve in the direction of the arrow (right). Slow down, keep right, and do not pass.',
        'The road you are traveling on intersects a highway ahead. Slow down, look to the right and to the left for other traffic, and be prepared to st',
        'Warns of hazardous condition on bridge caused by ice. This sign will be displayed continuously during winter time periods. Drivers should slow down, avoid applying brakes suddenly, or making sharp or sudden movements.',
        'The road or street ahead is for one-way traffic traveling in the opposite direction. You must not drive in that direction or else you will be driving into oncoming traffic.',
        'School Zone. The speed shown is in effect when the yellow light is flashing. Be extremely careful for school children.',
        'Railroad crossbuck signs are posted at every railroad, highway, road, or street grade crossing and show the location of the train tracks. If more than one track is to be crossed, the sign will show the number of tracks. Always slow down, look, listen, and be prepared to yield the right-of-way to an approaching train.',
        'Slow down, the road surface ahead is in poor condition.',
        'A flag person is often provided in roadway work zones to stop, slow, or guide traffic safely through the area. A flag person wears an orange vest, shirt, or jacket and uses stop/slow paddles or red flags to direct traffic through work zones.',
        'a. Slow down and increase the following distance when the road is wet. Many drivers find out too late what a small amount of rain can do. Roads become slippery when wet, making your car harder to control. Slow down and make sure you have complete control of the situation at all times.',
        'A red stop sign with white letters or a yellow sign with black letters. The stop sign means come to a complete stop, yield to pedestrians or other vehicles, and then proceed carefully. Stop before the crosswalk, intersection, or stop sign. This applies to each vehicle that comes to the sign. Slowing down is not adequate.',
        'If you are stopped behind a truck on an upgrade, leave space in case the truck drifts back when it starts to move. Also, keep to the left in your lane so the driver can see you’re stopped behind the truck.',
        'You need to be able to recognize other drivers who are engaged in any form of driving distraction. Not recognizing other distracted drivers can prevent you from perceiving or reacting correctly in time to prevent a crash. Watch for: • Vehicles that may drift over the lane divider lines or within their own lane. • Vehicles traveling at inconsistent speeds.• Drivers who are preoccupied with maps, food, cigarettes, cell phones, or other objects. • Drivers who appear to be involved in conversations with their passengers. Give a distracted driver plenty of room and maintain your safe following distance. Be very careful when passing a driver who seems to be distracted. The other driver may not be aware of your presence, and they may drift in front of you.',
        'You are near a school. Slow down, watch for children, and prepare to stop suddenly if necessary.',
        'The surface of the road is covered with loose gravel. Go slow enough to keep complete control of your vehicle. Do not apply brakes suddenly or make sharp turn',
        'A solid yellow line on your side of the road marks a “no-passing zone.” Broken or dashed lines permit you to pass or change lanes, if safe.',
        ' Various traffic control devices are used in construction and maintenance work areas to direct drivers, bicyclists, or pedestrians safely through the work zone and to provide for the safety of the workers. The most commonly used traffic control devices are signs, barricades, vertical panels, drums, cones, tubes, flashing arrow panels, and flag individuals. Orange is the basic color for these devices. When you are in a construction and maintenance work area, be prepared: 1. To slow down or stop as you approach workers and equipment 2. To change lanes 3. For unexpected movements of workers and equipment.',
        'Large flashing or sequencing arrow panels may be used in work zones day and night to guide drivers into certain traffic lanes and to inform them part of the road ahead is closed.',
        'Be on the lookout for cyclists on the road, especially at intersections. The most common car-bicycle crashes caused by a motorist are: 1. A motorist turns left in front of oncoming bicycle traffic. Oncoming bicycle traffic is often overlooked or its speed misjudged.2. A motorist turns right across the path of the bicycle. The motorist should slow down and merge with the bicycle traffic for a safe right turn. A motorist pulls away from a stop sign and fails to yield the right-of-way to bicycle cross traffic. At intersections, the right-ofway rules apply equally to motor vehicles and bicycles.',
        'Height of underpass from road surface is shown. Do not try to enter if your load is higher than the figure shown on the sign.',
        'Steady Green Light (Go) A steady green light means the driver can proceed on a green light if it is safe to do so. You may drive straight ahead or turn unless prohibited by another sign or signal. Watch for cars and pedestrians in the intersection. Be aware of reckless drivers who may race across the intersection to beat a red light.',
        'A red stop sign with white letters or a yellow sign with black letters. The stop sign means come to a complete stop, yield to pedestrians or other vehicles, and then proceed carefully. Stop before the crosswalk, intersection, or stop sign. This applies to each vehicle that comes to the sign. Slowing down is not adequate.',
        'Slow down and watch for individuals who may be disabled or who may be crossing the road in a wheelchair.',
        'You are approaching a downgrade; all drivers approach with caution. It may be necessary to use a lower gear to slow your vehicle.',
        'Do Not Cross Yellow Lines: The distance you can see ahead is so limited that passing another vehicle is hazardous and you may not pass.',
        'Railroad Crossing sign means you are within a few hundred feet of a railroad crossing. Slow down and be prepared to stop. If you see a train coming, STOP. Never try to beat a train',
        'Construction and maintenance signs are used to alert drivers of unusual or potentially dangerous conditions in or near work areas. Most signs in work areas are diamond shaped, but a few are rectangular.When you encounter any type of channelizing device: 1. Slow down and prepare to change lanes when it is safe to do so.\xa0 2. Be prepared for drivers who wait until the last second to move to the open lane. 3. Maintain reduced speed until you clear the construction area. There should be a sign indicating you are leaving the construction area. 4. Return to the normal driving lane only after checking traffic behind you.\xa0',
        'The bridge ahead is not as wide as the road. Slow down and use caution.',
        'There is a sudden high place in the road ahead. Slow down in order to avoid losing control of your vehicle or an uncomfortable jolt.',
        'DO NOT PASS sign:Do not pass other vehicles.',
        'If you see W0WRONG WAY sign facing you, you are driving the wrong way on a one-way street and you are directly opposing the flow of traffic.',
        'Since there are not any trafficcontrols at this intersection, make sure there are no approaching vehicles from the left. When approaching this type of intersection, yield the right-of-way to any vehicle that has entered or is approaching the intersection on your right. If the road to your right is clear or if approaching vehicles are far enough from the intersection to make your crossing safe, you may proceed. ',
        'When entering or crossing a road, street, or highway from a private road, alley, building, or driveway, you must stop prior to the sidewalk and yield the right-of-way to all approaching vehicles and pedestrians',
        'Steady Red Light (Stop) Stop before entering the crosswalk or intersection. You may turn right unless prohibited by law. You may also turn left if both streets are one way, unless prohibited by law. You must yield to all pedestrians and other traffic lawfully using the intersection',
        'Height of underpass from road surface is shown. Do not try to enter if your load is higher than the figure shown on the sign. ',
        'When you are in a construction and maintenance work area, be prepared: 1. To slow down or stop as you approach workers and equipment 2. To change lanes 3. For unexpected movements of workers and equipment',
        'Slow your speed and watch for trucks entering or crossing the road or highway.',
        'The hard-surfaced pavement changes to an earth road or low-type surface. Slow down.',
        'Ramp Metered When Flashing sign: The sign will have yellow lights flashing (top and bottom) when the freeway ramp ahead is metered. The ramp meter (red or green) directs motorists when to enter the freeway.',
        'The road you are traveling on intersects a highway ahead. Slow down, look to the right and to the left for other traffic, and be prepared to stop.',
        ' The lane ends ahead. If you are driving in the right lane, you should merge into the left.',
        'EXIT 25 MPH sign:Indicates the speed at which the exit ramp from a highway may be traveled safely.',
        'SPEED LIMIT 55 sign:Indicates the speed at which the exit ramp from a highway may be traveled safely.',
        'YIELD sign: This signs tells you the road you are on joins with another road ahead. You should slow down or stop if necessary so you can yield the right-of-way to vehicles, pedestrians, or bicycles on the other road.',
        'PROTECTED LEFT ON GREEN ARROW sign: Vehicles facing the signal with the green arrow may proceed safely into the intersection. While turning left, you are protected from oncoming traffic that must stop for vehicles at an intersection. Vehicles turning at a protected light should use caution.',
        'School Speed Limit 20 When Flashing sign: The use of a wireless communication device is prohibited in the school zone. School Speed Limit 20 When Flashing.',
        'A green signal will indicate when you may turn left.',
        'RIGHT LANE MUST TURN RIGHT sign: Vehicles driving in the right lane must turn right at the next intersection unless the sign indicates a different turning point.',
        'Respect a Motorcycle Allow the motorcyclist a full lane width. Although it may seem as though there is enough room in the traffic lane for an automobile and a motorcycle, the motorcycle is entitled to a full lane and may need the room to maneuver safely. Do not attempt to share the lane with a motorcycle.',
        'Slow down when driving at night and be sure you can stop within the distance lit by your headlights.',
        'The driver traveling on a frontage road of a controlled-access highway must yield the right-of-way to a vehicle: • Entering or about to enter the frontage road from the highway; and • Leaving or about to leave the frontage road to enter the highway.',
        'When approaching an intersection of a through street traveling from a street that ends at the intersection, you must stop and yield the right-of-way to vehicles on the through street.',
        'You must yield the right-of-way to school buses. Always drive with care when you are near a school bus. If you approach a school bus from either direction and the bus is displaying alternately flashing red lights, you must stop. Do not pass the school bus until: 1. The school bus has resumed motion; 2. You are signaled by the driver to proceed; or 3. The red lights are no longer flashing.',
        'When you are in a construction and maintenance work area, be prepared: 1. To slow down or stop as you approach workers and equipment 2. To change lanes 3. For unexpected movements of workers and equipment',
        'Steady Yellow Light (Caution) A steady yellow light warns drivers to use caution and to alert them the light is about to change to red. You must STOP before entering the nearest crosswalk at the intersection if you can do so safely. If a stop cannot be made safely, then you may proceed cautiously through the intersection before the light changes to red.',
        'Reverse Turn sign: The road curves one way (right) and then the other way (left). Slow down, keep right, and do not pass.',
        'Right Turn Ahead sign: Road ahead makes a sharp turn in the direction of the arrow (right). Slow down, keep right, and do not pass.',
        'White stop lines are painted across the pavement lanes at traffic signs or signals. Where these lines are present, you are required to stop behind the stop line.',
        'Barricades, vertical panels, drums, cones, and tubes are the most commonly used devices to alert drivers of unusual or potentially dangerous conditions in highway and street work areas, and to guide drivers safely through the work zone. At night channelizing devices are often equipped with flashing or steady burn lights.',
        'Two feet of rushing water will carry away pick-up trucks, SUVs, and most other vehicles.• Water across a road may hide a missing segment of roadbed or a missing bridge. Roads weaken under floodwater and drivers should proceed cautiously after waters have receded since the road may collapse under the vehicle’s weight',
        'Respect a Motorcycle Allow the motorcyclist a full lane width. Although it may seem as though there is enough room in the traffic lane for an automobile and a motorcycle, the motorcycle is entitled to a full lane and may need the room to maneuver safely. Do not attempt to share the lane with a motorcycle.',
        'On roads where there’s a speed limit sign, you must not drive faster than that speed limit.',
    ]


    NSW = [
        'At any time when you are travelling in the same direction as a bus with a ‘40 when lights flash’ sign on the back and the lights on top are flashing, you must not overtake it at more than 40km/h. This is because the bus is picking up or dropping off children who may be crossing or about to cross the road.',
        'Even if you’re driving at or below the speed limit, you may be driving too fast for road conditions such as curves, rain, heavy traffic or night-time.',
        'Older people Older people may be slower than other pedestrians and may not see you until you’re very close. Slow down and give them extra time to cross.',
        'You must stop at a ‘Stop’ sign held by a traffic controller, for example, at roadworks and children’s crossings. You must remain at a complete stop until the controller stops showing the sign or signals you can go.',
        'When you see the ‘No entry’ sign, you must not turn into or enter the road.',
        'Regulatory speed sign: Speed limit signs show you the maximum speed you can drive in good conditions. Slow down in poor conditions. You must not drive faster than the speed limit shown on the sign.',
        'Local traffic areas sign: A local traffic area is an area of local streets with a speed limit of 40km/h.',
        'Shared zone sign: Shared zones have a speed limit of 10km/h. You must not drive faster than this speed limit. You must also give way to any pedestrian in a shared zone. This includes slowing down and stopping, if necessary, to avoid them.',
        'common crash types that should be avoided: Colliding with another vehicle coming from an adjacent direction (the left or right).',
        'You should increase your crash avoidance space to 4 or more seconds when driving in poor conditions, such as on unsealed (dirt or gravel), icy or wet roads, or at night.',
        'The rules You must keep enough distance between you and the vehicle travelling in front so you can, if necessary, stop safely to avoid colliding with the vehicle.',
        'Road work speed limit sign: Roadwork signs alert you to the start and end of roadworks and the speed limit for that area. You must not go faster than the speed limit shown on the sign.',
        'Common crash types that should be avoided: Running off the road on a straight section and hitting an object or parked vehicle.',
        'Be aware of who you’re sharing the road with and how you can take care around them. Allow enough time to stop safely for pedestrians. Give other vehicles enough room to stop and turn. Keep an eye out for bicycle and motorcycle riders.',
        'Wheelchair Crossing Sign: Slow down and watch for individuals who may be disabled or who may be crossing the road in a wheelchair.',
        'School zone sign: You must not drive faster than the speed limit in a school zone on school days during the times shown on the sign. School days are published by the NSW Department of Education.',
        'Default speed limits apply on roads without speed limit signs or roads with an end speed limit sign. 1.End speed limit sign: 60km/h. 2. State limit sign 100km/h',
        'You must not drive a vehicle on a road negligently or at a speed or in a manner dangerous to the public. You must not drive a vehicle on a road negligently or at a speed or in a manner dangerous to the public.',
        'ommon crash types that should be avoided: Colliding with the rear of another vehicle (rear-end).',
        'As a driver, it’s your responsibility to help keep motorcycle riders safe. Do not drive alongside Do not drive alongside and in the same lane as a motorcycle. They have a right to a full-width lane to ride safely.',
        'Do not cut in front of a truck or bus. Give them enough room to stop safely.',
        'Slow down for buses with flashing lights If you’re travelling in the same direction as a bus with a ‘40 when lights flash’ sign on the back , you must not overtake it at more than 40km/h while the lights on top are flashing. This is because the bus is picking up or dropping off children.',
        'Give way sign: When you approach a ‘Give way’ sign you must slow down and prepare to stop.',
        "Floodwater is extremely dangerous. Find another way or wait until the road is clear. It's safer to turn around than to drive in floodwater.",
        'Before you drive, check for storms, bushfires, hail, snow, dust storms and heavy fog. When you cannot avoid driving in poor conditions, slow down, drive carefully and increase your visibility by using your day running lights or headlights.',
        "If it starts to rain, you should turn on your headlights (if they don't come on automatically), break gently to slow down, and increase the gap between you and the vehicle in front (crash avoidance space).",
        'Funeral processions When you see a funeral or an official procession, you must not interrupt it. You can get a fine if you interfere with the procession.',
        'When passing a stopped emergency vehicle with flashing blue or red lights: • If the speed limit is 80km/h or less, you must slow down to 40km/h.',
        'When passing a stopped tow truck or breakdown assistance vehicle with flashing lights: • If the speed limit is 80km/h or less, you must slow down to 40km/h.',
        "Crashes If you're involved in a crash you must always stop and give as much help as possible. You must provide your details to the other people involved or to police.",
        'A temporary arrow on a roadwork vehicle warns you that a road hazard is ahead. Change lanes.',
        'Traffic controller ahead sign: Be prepared to stop',
        'You must obey the regulatory signs and traffic lights at roadworks. Look out for road workers on the road and obey signals from traffic controllers.',
        'Some roads have large electronic signs (called variable message signs). These signs warn you of changes in traffic conditions ahead – for example, fog, a crash, roadworks, congestion, road closures or police operations.',
        'While driving, look out for potential hazards. A hazard is any possible danger that might lead to a crash. It could be a pedestrian waiting to cross, a wet road, or something blocking your view of oncoming vehicles. Also look out for approaching vehicles and parked vehicles pulling out.',
        "When a sign or lane markings show 'Bus only' or 'Buses only', only buses can drive in these lanes.",
        'Speed limit signals: Some motorways have overhead electronic speed limit signs (called variable speed limit signs) that show the speed limit. You must not drive over the speed limit shown.',
        "When there's a 'No overtaking or passing' sign on a bridge, you: • must give way to vehicles approaching in the opposite direction • must not overtake any vehicle travelling in the same direction.",
        "When there's a 'Give way' sign at a level crossing, you must slow down, look both ways and stop if a train is coming.",
        'Children may be crossing ahead sign: These areas may have a lower speed limit and signs warning you to look out for pedestrians. When you see these signs, you should prepare to slow down.',
        "When a children's crossing is operating it's marked by red‑orange flags at both sides. When you see the flags, you must slow down and stop before the white stripes or 'Stop' line to give way to pedestrians. You must remain at a complete stop until all pedestrians have left the crossing.",
        "More than half of all pedestrian fatalities occur in darkness or at dusk. Slow down and prepare to stop when visibility is poor, for example, in rain or fog, or at night, dawn or dusk. Pedestrians are harder to see and they're also more likely to hurry and take risk",
        'Drivers should look out for mobility scooters or motorised wheelchairs. Take particular care when entering or leaving a driveway, as they can be difficult to see and move faster than other pedestrians.',
        "Bicycle riders are vulnerable users and do not have the same protections as people in vehicles and can be seriously injured or killed in a crash. As a driver, it's your responsibility to help keep bicycle riders safe.",
        "Do not put your vehicle in the path of a large, heavy vehicle when it's turning – you may be crushed. It's safest to stay behind and wait until the vehicle has completed the turn. There are also rules that must be obeyed.",
        'Look out for motorcycle riders. More than half of all motorcycle crashes involving other vehicles happen at intersections.',
        'As a driver, you must take care to avoid colliding with bicycles turning at intersections.',
        'When being overtaken, you should: • stay in your lane • keep left • allow room for the overtaking vehicle to pass and move back into the lane.',
        'Driving on unsealed roads Take extra care and slow down when driving on unsealed roads (dirt or gravel). Your vehicle takes longer to stop and is harder to control. If you drive too fast, your vehicle may skid, slide or roll over.',
        "Driving through water You should avoid driving through water. It can be very risky. There's a limit to the depth of water that your vehicle can drive through safely.",
        'If you see an animal on or near the road, slow down and apply your brakes in a controlled way. Never swerve to avoid an animal. This may cause you to lose control of your vehicle or to collide with oncoming traffic.',
    ]

    Japan = [
        'When the maximum or minimum speed is specified with road signs and displays, you must not exceed or drive slower than that speed limit.',
        'Flashing red light: Vehicles and streetcars/trams must stop at the stopping point before proceeding.',
        'Flashing yellow light: Pedestrians, vehicles and streetcars/trams may proceed carefully, paying attention to other traffic.',
        'slow sign: Vehicles must drive at a speed where it is possible to stop immediately.',
        'stop sign: Vehicles must stop just before a stop line or intersection.* When doing so, vehicles must not obstruct traffic at the intersecting roads.',
        'Drivers must yield to pedestrians.',
        'Prohibition for Motor Vehicles sign: Command or Prohibition Prohibition for motor vehicles and other multi-track motor vehicles.',
        'Warning signs call for increased attention, especially for reducing speed in anticipation of a hazardous situation.',
        'When passengers are boarding or alighting, vehicles may only pass on the right at walking speed and at such a distance that passenger safety is not jeopardized. Passengers must not be hindered. If necessary, drivers must wait.',
        'Other vehicles must allow public transport buses and school buses to depart from designated stops. If necessary, other vehicles must wait.',
        'Start of a 30 km/h Zone" Sign: Command or Prohibition Anyone driving a vehicle must not exceed the maximum speed limit indicated within this zone. Explanation: Along with this sign, speed limits of less than 30 km/h can be imposed in traffic-calmed business areas. This helps ensure that traffic moves slowly and safely through areas with high pedestrian activity or where vehicles frequently interact with other road users.',
    ]

    German = [
        'If visibility is reduced to less than 50 meters due to fog, snowfall, or rain, the speed must not exceed 50 km/h, unless a lower speed is required.',
        'Anyone wishing to turn left must allow oncoming vehicles that want to turn right to pass. Oncoming vehicles wanting to turn left must turn in front of each other, unless the traffic situation or the design of the intersection requires waiting until the vehicles have passed each other.',
        "Vehicles must wait before the St. Andrew's cross and pedestrians at a safe distance before the railroad crossing when:\nA rail vehicle is approaching,Red flashing light or yellow or red light signals are given,\nThe barriers are lowering or are closed,\nA railway employee signals to stop,\nAn audible signal, such as a whistle from an approaching train, is heard.\nIf the red flashing light or red light signal is in the form of an arrow, only those intending to drive in the direction of the arrow must wait. The lowering of the barriers may be announced by a bell signal.",
        'At bus stops (Sign 224) where public transport buses, trams, and designated school buses are stopping, vehicles, including those in the oncoming traffic, may only pass cautiously.',
        'Maximum Speed Limit Sign: Command or Prohibition A person driving a vehicle must not exceed the speed limit indicated on the sign.',
        'No Entry" Sign: A person driving a vehicle is not permitted to enter the roadway for which the sign is designated. Explanation: The sign is positioned on the right side of the roadway to which it applies, or on both sides of that roadway.',
        'Where there are two or more motorized lanes in the same direction on the road, the left side is the fast lane and the right side is the slow lane. Motor vehicles traveling in a fast lane shall drive at the speed specified in the fast lane, and those that have not reached the speed specified in the fast lane shall drive in a slow lane. Motorcycles should drive in the rightmost lane. If there are traffic signs indicating the driving speed, drive at the indicated driving speed. When a motor vehicle in a slow lane overtakes the preceding vehicle, it can borrow the fast lane to drive. Where there are two or more motor vehicle lanes in the same direction on the road, the motor vehicle that changes lanes shall not affect the normal driving of the motor vehicle in the relevant lane.',
    ]

    China = [
        'Lane signal lights indicate: (1) When the green arrow light is on, allow vehicles in the lane to pass in the direction indicated; (2) When the red cross-shaped light or arrow light is on, vehicles in the lane are prohibited from passing.',
        'Motor vehicles must not exceed the speed indicated by the speed limit signs and markings on the road. On roads without speed limit signs and markings, motor vehicles shall not exceed the following maximum speeds. (1) For roads without a road centerline, urban roads are 30 kilometers per hour, and highways are 40 kilometers per hour; (2) For roads with only one motor vehicle lane in the same direction, 50 kilometers per hour for urban roads and 70 kilometers per hour for highways.',
        'On roads without central isolation facilities or without a central line, motor vehicles come in opposite directions. The following regulations should be observed when driving: (1) Slow down and keep to the right, and keep a necessary safe distance from other vehicles and pedestrians; (2) On a road with obstacles, the side with obstacles shall go first; but when the side with obstacles has entered the road with obstacles and the side with obstacles has not, the side with obstacles shall go first; (3) On a narrow slope, the uphill side goes first; but when the downhill side has reached halfway and the uphill side is not uphill, the downhill side goes first; (4) On the narrow mountain road, the side that does not rely on the mountain shall go first; (5) At night meeting vehicles should switch to low beam lights 150 meters away from the oncoming vehicle in the opposite direction, and should use low beam lights when meeting vehicles on narrow roads, narrow bridges and non-motorized vehicles.',
        'Motor vehicles passing through intersections controlled by traffic lights shall pass in accordance with the following regulations: (1) At an intersection with a guide lane, drive into the guide lane according to the required direction of travel; (2) Those who are preparing to enter the roundabout let motor vehicles already in the intersection go ahead; (3) When turning to the left, turn to the left of the center of the intersection. Turn on the turn signal when turning, and turn on the low beam when driving at night; (4) Pass in turn when encountering a release signal; (5) When the stop signal is encountered, stop outside the stop line in turn. If there is no stop line, stop outside the intersection; (6) When turning right when there is a car in the same lane waiting for the release signal, stop and wait in turn; (7) At intersections with no direction indicator lights, turning motor vehicles let straight vehicles and pedestrians go first. Right-turning motor vehicles traveling in the opposite direction let left-turning vehicles go first.',
        'The flashing warning signal light is a yellow light that continues to flash, reminding vehicles and pedestrians to pay attention when passing through, and pass after confirming safety.',
        'When two red lights flash alternately or one red light is on a road and railway intersection, it means that vehicles and pedestrians are prohibited; when the red light is off, it means that vehicles and pedestrians are allowed to pass.',
    ]

    Choosed_Texas =[
        'When approaching a railroad grade crossing, stop between 15 and 50 feet from the nearest rail if: 1. A clearly visible railroad signal warns of an approaching train 2. A crossing gate is lowered or a flag person warns of an approaching train 3. A driver is required to stop by an official traffic-control device or a traffic-control signal 4. An approaching train is within about 1,500 feet of the crossing. The train will produce an audible signal to identify the immediate hazard. 6. An approaching train is visible and in close proximity to the crossing',
        'A Flashing Red Light Stop completely before entering the crosswalk or intersection, then proceed when you can do so safely. Vehicles on the intersecting road may not have to stop',
        'You must yield the right-of-way to police cars, fire trucks, ambulances, and other emergency vehicles sounding a siren, bell, or flashing red light. If traffic allows, pull to the right edge of the road and stop. If you are unable to pull over to the right, slow down and leave a clear path for the emergency vehicle.',
        'Slow down on wet road. Do not suddenly turn, speed up, or stop.',
        'Before you drive, check for storms, bushfires, hail, snow, dust storms and heavy fog. When you cannot avoid driving in poor conditions, slow down, drive carefully and increase your visibility by using your day running lights or headlights.',
        'Slow down when driving at night and be sure you can stop within the distance lit by your headlights.',
        'Warns of hazardous condition on bridge caused by ice. This sign will be displayed continuously during winter time periods. Drivers should slow down, avoid applying brakes suddenly, or making sharp or sudden movements.',
        'The road or street ahead is for one-way traffic traveling in the opposite direction. You must not drive in that direction or else you will be driving into oncoming traffic.',
        'School Zone. The speed shown is in effect when the yellow light is flashing. Be extremely careful for school children.',
        'Railroad crossbuck signs are posted at every railroad, highway, road, or street grade crossing and show the location of the train tracks. If more than one track is to be crossed, the sign will show the number of tracks. Always slow down, look, listen, and be prepared to yield the right-of-way to an approaching train.',
        'A flag person is often provided in roadway work zones to stop, slow, or guide traffic safely through the area. A flag person wears an orange vest, shirt, or jacket and uses stop/slow paddles or red flags to direct traffic through work zones.',
        'a. Slow down and increase the following distance when the road is wet. Many drivers find out too late what a small amount of rain can do. Roads become slippery when wet, making your car harder to control. Slow down and make sure you have complete control of the situation at all times.',
        'A red stop sign with white letters or a yellow sign with black letters. The stop sign means come to a complete stop, yield to pedestrians or other vehicles, and then proceed carefully. Stop before the crosswalk, intersection, or stop sign. This applies to each vehicle that comes to the sign. Slowing down is not adequate.',
        'You need to be able to recognize other drivers who are engaged in any form of driving distraction. Not recognizing other distracted drivers can prevent you from perceiving or reacting correctly in time to prevent a crash. Watch for: • Vehicles that may drift over the lane divider lines or within their own lane. • Vehicles traveling at inconsistent speeds.• Drivers who are preoccupied with maps, food, cigarettes, cell phones, or other objects. • Drivers who appear to be involved in conversations with their passengers. Give a distracted driver plenty of room and maintain your safe following distance. Be very careful when passing a driver who seems to be distracted. The other driver may not be aware of your presence, and they may drift in front of you.',
        'The surface of the road is covered with loose gravel. Go slow enough to keep complete control of your vehicle. Do not apply brakes suddenly or make sharp turn',
        'Various traffic control devices are used in construction and maintenance work areas to direct drivers, bicyclists, or pedestrians safely through the work zone and to provide for the safety of the workers. The most commonly used traffic control devices are signs, barricades, vertical panels, drums, cones, tubes, flashing arrow panels, and flag individuals. Orange is the basic color for these devices. When you are in a construction and maintenance work area, be prepared: 1. To slow down or stop as you approach workers and equipment 2. To change lanes 3. For unexpected movements of workers and equipment.',
        'Be on the lookout for cyclists on the road, especially at intersections. The most common car-bicycle crashes caused by a motorist are: 1. A motorist turns left in front of oncoming bicycle traffic. Oncoming bicycle traffic is often overlooked or its speed misjudged.2. A motorist turns right across the path of the bicycle. The motorist should slow down and merge with the bicycle traffic for a safe right turn. A motorist pulls away from a stop sign and fails to yield the right-of-way to bicycle cross traffic. At intersections, the right-ofway rules apply equally to motor vehicles and bicycles.',
        'Height of underpass from road surface is shown. Do not try to enter if your load is higher than the figure shown on the sign.',
        'Slow down and watch for individuals who may be disabled or who may be crossing the road in a wheelchair.',
        'You are approaching a downgrade; all drivers approach with caution. It may be necessary to use a lower gear to slow your vehicle.',
        'Railroad Crossing sign means you are within a few hundred feet of a railroad crossing. Slow down and be prepared to stop. If you see a train coming, STOP. Never try to beat a train',
        'Construction and maintenance signs are used to alert drivers of unusual or potentially dangerous conditions in or near work areas. Most signs in work areas are diamond shaped, but a few are rectangular.When you encounter any type of channelizing device: 1. Slow down and prepare to change lanes when it is safe to do so.\xa0 2. Be prepared for drivers who wait until the last second to move to the open lane. 3. Maintain reduced speed until you clear the construction area. There should be a sign indicating you are leaving the construction area. 4. Return to the normal driving lane only after checking traffic behind you.\xa0',
        'There is a sudden high place in the road ahead. Slow down in order to avoid losing control of your vehicle or an uncomfortable jolt.',
        'If you see W0WRONG WAY sign facing you, you are driving the wrong way on a one-way street and you are directly opposing the flow of traffic.',
        'Slow your speed and watch for trucks entering or crossing the road or highway.',
        'The hard-surfaced pavement changes to an earth road or low-type surface. Slow down.',
        'The road you are traveling on intersects a highway ahead. Slow down, look to the right and to the left for other traffic, and be prepared to stop.',
        'EXIT 25 MPH sign:Indicates the speed at which the exit ramp from a highway may be traveled safely.',
        'YIELD sign: This signs tells you the road you are on joins with another road ahead. You should slow down or stop if necessary so you can yield the right-of-way to vehicles, pedestrians, or bicycles on the other road.',
        'School Speed Limit 20 When Flashing sign: The use of a wireless communication device is prohibited in the school zone. School Speed Limit 20 When Flashing.',
        'A green signal will indicate when you may turn left.',
        'RIGHT LANE MUST TURN RIGHT sign: Vehicles driving in the right lane must turn right at the next intersection unless the sign indicates a different turning point.',
        'Respect a Motorcycle Allow the motorcyclist a full lane width. Although it may seem as though there is enough room in the traffic lane for an automobile and a motorcycle, the motorcycle is entitled to a full lane and may need the room to maneuver safely. Do not attempt to share the lane with a motorcycle.',
        'The driver traveling on a frontage road of a controlled-access highway must yield the right-of-way to a vehicle: • Entering or about to enter the frontage road from the highway; and • Leaving or about to leave the frontage road to enter the highway.',
        'You must yield the right-of-way to school buses. Always drive with care when you are near a school bus. If you approach a school bus from either direction and the bus is displaying alternately flashing red lights, you must stop. Do not pass the school bus until: 1. The school bus has resumed motion; 2. You are signaled by the driver to proceed; or 3. The red lights are no longer flashing.',
        'When you are in a construction and maintenance work area, be prepared: 1. To slow down or stop as you approach workers and equipment 2. To change lanes 3. For unexpected movements of workers and equipment',
        'Steady Yellow Light (Caution) A steady yellow light warns drivers to use caution and to alert them the light is about to change to red. You must STOP before entering the nearest crosswalk at the intersection if you can do so safely. If a stop cannot be made safely, then you may proceed cautiously through the intersection before the light changes to red.',
        'Reverse Turn sign: The road curves one way (right) and then the other way (left). Slow down, keep right, and do not pass.',
        'Right Turn Ahead sign: Road ahead makes a sharp turn in the direction of the arrow (right). Slow down, keep right, and do not pass.',
        'White stop lines are painted across the pavement lanes at traffic signs or signals. Where these lines are present, you are required to stop behind the stop line.',
        'Barricades, vertical panels, drums, cones, and tubes are the most commonly used devices to alert drivers of unusual or potentially dangerous conditions in highway and street work areas, and to guide drivers safely through the work zone. At night channelizing devices are often equipped with flashing or steady burn lights.',
        'Two feet of rushing water will carry away pick-up trucks, SUVs, and most other vehicles.• Water across a road may hide a missing segment of roadbed or a missing bridge. Roads weaken under floodwater and drivers should proceed cautiously after waters have receded since the road may collapse under the vehicle’s weight',
        'On roads where there’s a speed limit sign, you must not drive faster than that speed limit.',
    ]

    LLM = MRGenerator()
    data_all = Choosed_Texas
    idx = 0
    data_lists = []
    idx_1 = 0
    for item in tqdm(data_all, desc="Processing items"):
        idx += 1
        MRs_, prompt, categorys_, road_network_, object_environments_, maneuver_, diffusion_prompts,score = LLM(item)
        torch.cuda.empty_cache()
        for i in range(len(object_environments_)):
            torch.cuda.empty_cache()
            category = categorys_[i]
            road_network = road_network_
            maneuver = maneuver_
            object_environment = object_environments_[i]
            diffusion_prompt = diffusion_prompts[i]
            MR = MRs_[i]
            if category=="weather":
                type="Pix2Pix"
            else:
                type = "diffusion"
            data = {
                #"idx":idx,
                "idx_new": idx_1,
                #"prompt": prompt,
                #"road_network": road_network,
               # "objects_environment": object_environment,
                "maneuver": maneuver,
                #"category": category,
                #"MR": MR,
                #"score": score,
                "diffusion_prompt": diffusion_prompt,
                "type":type
            }
            data_lists.append(data)
            idx_1+=1
    with open(f"Texas_final.json", 'w', encoding='utf-8') as f:
        json.dump(data_lists, f, ensure_ascii=False, indent=4)

