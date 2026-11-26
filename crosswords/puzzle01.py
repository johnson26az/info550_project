# puzzle02.py

# a simplified crossword puzzle
# four copies and create valid four-letter across and down entries.
# trying to see if the node2vec agent can solve this simple crossword puzzle
#
# The word square is:
#
# B A L L
# A R E A
# L E A D
# L A D Y
#
# Its columns are the same four words as its rows.

crossword = {
    "size": (10, 10),

    "grid": [
        "    #    #",
        "    #    #",
        "    #    #",
        "    #    #",
        "##########",
        "    #    #",
        "    #    #",
        "    #    #",
        "    #    #",
        "##########"
    ],

    "clues": {
        "across": {
            1: "A round object used in sports or games",
            5: "A round object used in sports or games",
            9: "A region or space",
            10: "A region or space",
            11: "A substance made from plants, or to guide",
            12: "A substance made from plants, or to guide",
            13: "A young woman",
            14: "A young woman",
            15: "A round object used in sports or games",
            19: "A round object used in sports or games",
            23: "A region or space",
            24: "A region or space",
            25: "A substance made from plants, or to guide",
            26: "A substance made from plants, or to guide",
            27: "A young woman",
            28: "A young woman"
        },
        "down": {
            1: "A round object used in sports or games",
            2: "A region or space",
            3: "A substance made from plants, or to guide",
            4: "A young woman",
            5: "A round object used in sports or games",
            6: "A region or space",
            7: "A substance made from plants, or to guide",
            8: "A young woman",
            15: "A round object used in sports or games",
            16: "A region or space",
            17: "A substance made from plants, or to guide",
            18: "A young woman",
            19: "A round object used in sports or games",
            20: "A region or space",
            21: "A substance made from plants, or to guide",
            22: "A young woman"
        }
    },

    "word_bank": [
        "BALL", "AREA","LEAD", "LADY", "ABLE",
        "ACID", "AGED", "ALSO", "ARCH", "ARMY",
        "AWAY", "BABY", "BACK", "BAKE", "BAND",
        "BANK", "BASE", "BATH", "BEAR", "BEAT",
        "BELL", "BELT", "BEST", "BIKE", "BIRD",
        "BALL", "AREA","LEAD", "LADY", "ABLE",
        "BLOW", "BLUE", "BOAT", "BODY", "BOND",
        "BOOK", "BOOT", "BORN", "BOWL", "BREW",
        "CAKE", "CALL", "CALM", "CAME", "CAMP",
        "CARD", "CARE", "CASE", "CASH", "CHAT",
        "CITY", "CLAY", "CLIP", "CLUB", "COAT",
        "BALL", "AREA","LEAD", "LADY", "ABLE",
        "CODE", "COLD", "COME", "COOK", "COOL",
        "COPY", "DAWN", "DEAL", "DEEP", "DESK",
        "EAST", "ECHO", "EDGE", "ELSE", "EDIT",
        "BALL", "AREA","LEAD", "LADY", "ABLE",
        "EVER", "FACE", "FACT", "FAIL", "FAIR",
        "FALL", "FARM", "FAST", "FEAR", "FEEL",
        "FILE", "FILL", "FIRE", "FISH", "FILM"
    ],

    "solution_grid": [
        "BALL#BALL#",
        "AREA#AREA#",
        "LEAD#LEAD#",
        "LADY#LADY#",
        "##########",
        "BALL#BALL#",
        "AREA#AREA#",
        "LEAD#LEAD#",
        "LADY#LADY#",
        "##########"
    ]
}
