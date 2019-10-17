# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 18:07:48 2019

@author: Flo
"""

from random import choice

import wikipedia
wikipedia.set_lang("DE")


prompt = "Wien"
#def fragDenOnkel(prompt):

thema = wikipedia.search(prompt)[1] # <- der onkel glaubt python indiziert ab 1

artikel = wikipedia.page(thema)

openings = [f"Also, du willst also etwas über {thema} schreiben?",
            f"{thema}, ja, gut.",
            ]

openings2 = [f"Worum gehts den bei {thema}?",
             ]

print(choice(openings))