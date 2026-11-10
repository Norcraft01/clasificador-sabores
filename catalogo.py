"""Catálogo único de platillos: sabores, nombre en español y búsquedas para descargar fotos.

Sabores (letras): A=ácido, M=amargo, D=dulce, P=picante, S=salado.

- FOOD101: los 101 platillos de Food-101 con la MISMA tabla de sabores del modelo original.
- EXTRA:   platillos nuevos (comida mexicana + refuerzos de amargo/ácido) que se descargan de internet.
- REFUERZOS: búsquedas de fotos mexicanas que se agregan a una clase que YA existe en Food-101
             (por ejemplo, tacos de taquería dentro de "tacos").
"""

SABORES = ["acido", "amargo", "dulce", "picante", "salado"]
LETRAS = {"A": "acido", "M": "amargo", "D": "dulce", "P": "picante", "S": "salado"}

# ---------------------------------------------------------------------------
# Food-101: tabla original (sin cambios) + nombre en español para la interfaz
# ---------------------------------------------------------------------------
FOOD101 = {
    "apple_pie": ("D", "Pay de manzana"), "baby_back_ribs": ("SD", "Costillas BBQ"),
    "baklava": ("D", "Baklava"), "beef_carpaccio": ("SA", "Carpaccio de res"),
    "beef_tartare": ("SA", "Tartar de res"), "beet_salad": ("DA", "Ensalada de betabel"),
    "beignets": ("D", "Beignets"), "bibimbap": ("SP", "Bibimbap"),
    "bread_pudding": ("D", "Budín de pan"), "breakfast_burrito": ("SP", "Burrito de desayuno"),
    "bruschetta": ("SA", "Bruschetta"), "caesar_salad": ("SA", "Ensalada César"),
    "cannoli": ("D", "Cannoli"), "caprese_salad": ("SA", "Ensalada caprese"),
    "carrot_cake": ("D", "Pastel de zanahoria"), "ceviche": ("AS", "Ceviche"),
    "cheesecake": ("D", "Pay de queso"), "cheese_plate": ("S", "Tabla de quesos"),
    "chicken_curry": ("SP", "Pollo al curry"), "chicken_quesadilla": ("S", "Quesadilla"),
    "chicken_wings": ("SP", "Alitas"), "chocolate_cake": ("DM", "Pastel de chocolate"),
    "chocolate_mousse": ("DM", "Mousse de chocolate"), "churros": ("D", "Churros"),
    "clam_chowder": ("S", "Crema de almejas"), "club_sandwich": ("S", "Club sándwich"),
    "crab_cakes": ("S", "Pastelitos de cangrejo"), "creme_brulee": ("D", "Crème brûlée"),
    "croque_madame": ("S", "Croque madame"), "cup_cakes": ("D", "Cupcakes"),
    "deviled_eggs": ("S", "Huevos rellenos"), "donuts": ("D", "Donas"),
    "dumplings": ("S", "Dumplings"), "edamame": ("S", "Edamame"),
    "eggs_benedict": ("S", "Huevos benedictinos"), "escargots": ("S", "Escargots"),
    "falafel": ("S", "Falafel"), "filet_mignon": ("S", "Filete mignon"),
    "fish_and_chips": ("S", "Fish and chips"), "foie_gras": ("S", "Foie gras"),
    "french_fries": ("S", "Papas a la francesa"), "french_onion_soup": ("S", "Sopa de cebolla"),
    "french_toast": ("D", "Pan francés"), "fried_calamari": ("S", "Calamares fritos"),
    "fried_rice": ("S", "Arroz frito"), "frozen_yogurt": ("DA", "Yogur helado"),
    "garlic_bread": ("S", "Pan de ajo"), "gnocchi": ("S", "Ñoquis"),
    "greek_salad": ("SA", "Ensalada griega"), "grilled_cheese_sandwich": ("S", "Sándwich de queso"),
    "grilled_salmon": ("S", "Salmón a la parrilla"), "guacamole": ("SA", "Guacamole"),
    "gyoza": ("S", "Gyoza"), "hamburger": ("S", "Hamburguesa"),
    "hot_and_sour_soup": ("SAP", "Sopa agripicante"), "hot_dog": ("S", "Hot dog"),
    "huevos_rancheros": ("SP", "Huevos rancheros"), "hummus": ("S", "Hummus"),
    "ice_cream": ("D", "Helado"), "lasagna": ("S", "Lasaña"),
    "lobster_bisque": ("S", "Bisque de langosta"), "lobster_roll_sandwich": ("S", "Rollo de langosta"),
    "macaroni_and_cheese": ("S", "Macarrones con queso"), "macarons": ("D", "Macarons"),
    "miso_soup": ("S", "Sopa miso"), "mussels": ("S", "Mejillones"),
    "nachos": ("SP", "Nachos"), "omelette": ("S", "Omelette"),
    "onion_rings": ("S", "Aros de cebolla"), "oysters": ("S", "Ostiones"),
    "pad_thai": ("SDA", "Pad thai"), "paella": ("S", "Paella"),
    "pancakes": ("D", "Hot cakes"), "panna_cotta": ("D", "Panna cotta"),
    "peking_duck": ("SD", "Pato pekín"), "pho": ("S", "Pho"),
    "pizza": ("S", "Pizza"), "pork_chop": ("S", "Chuleta de cerdo"),
    "poutine": ("S", "Poutine"), "prime_rib": ("S", "Prime rib"),
    "pulled_pork_sandwich": ("SD", "Sándwich de cerdo deshebrado"),
    "ramen": ("S", "Ramen"), "ravioli": ("S", "Ravioles"),
    "red_velvet_cake": ("D", "Pastel red velvet"), "risotto": ("S", "Risotto"),
    "samosa": ("SP", "Samosa"), "sashimi": ("S", "Sashimi"),
    "scallops": ("S", "Callos de hacha"), "seaweed_salad": ("SA", "Ensalada de algas"),
    "shrimp_and_grits": ("S", "Camarones con sémola"), "spaghetti_bolognese": ("S", "Espagueti boloñesa"),
    "spaghetti_carbonara": ("S", "Espagueti carbonara"), "spring_rolls": ("S", "Rollos primavera"),
    "steak": ("S", "Bistec"), "strawberry_shortcake": ("D", "Pastel de fresas"),
    "sushi": ("S", "Sushi"), "tacos": ("SP", "Tacos"),
    "takoyaki": ("S", "Takoyaki"), "tiramisu": ("DM", "Tiramisú"),
    "tuna_tartare": ("SA", "Tartar de atún"), "waffles": ("D", "Waffles"),
}

# ---------------------------------------------------------------------------
# Platillos nuevos: clase -> (sabores, nombre, [búsquedas de imágenes])
# ---------------------------------------------------------------------------
EXTRA = {
    # --- Comida mexicana salada / picante ---
    "enchiladas": ("SP", "Enchiladas", ["enchiladas verdes plato", "enchiladas rojas mexicanas", "enchiladas suizas"]),
    "chilaquiles": ("SP", "Chilaquiles", ["chilaquiles verdes", "chilaquiles rojos con huevo", "chilaquiles mexicanos plato"]),
    "pozole": ("SP", "Pozole", ["pozole rojo", "pozole blanco plato", "pozole verde guerrero"]),
    "tamales": ("S", "Tamales", ["tamales mexicanos", "tamal en hoja de maíz abierto", "tamales oaxaqueños hoja de plátano"]),
    "mole_poblano": ("SDPM", "Mole poblano", ["mole poblano con pollo", "mole negro oaxaqueño plato", "enmoladas"]),
    "elote": ("SPA", "Elote preparado", ["elote preparado con mayonesa y chile", "elote asado con chile y limón"]),
    "esquites": ("SPA", "Esquites", ["esquites en vaso", "esquites con chile y limón"]),
    "sopes": ("SP", "Sopes", ["sopes mexicanos", "sopes de frijol con crema y queso"]),
    "tostadas": ("SP", "Tostadas", ["tostadas de tinga", "tostadas de pollo mexicanas", "tostadas de pata"]),
    "gorditas": ("SP", "Gorditas", ["gorditas de chicharrón", "gorditas mexicanas rellenas"]),
    "torta_ahogada": ("SPA", "Torta ahogada", ["torta ahogada jalisco", "torta ahogada en salsa"]),
    "pambazo": ("SP", "Pambazo", ["pambazo mexicano", "pambazos con papa y chorizo"]),
    "tlayuda": ("S", "Tlayuda", ["tlayuda oaxaqueña", "tlayuda con tasajo"]),
    "birria": ("SP", "Birria", ["birria de res consomé", "birria de chivo plato", "quesabirria tacos"]),
    "cochinita_pibil": ("SA", "Cochinita pibil", ["cochinita pibil con cebolla morada", "tacos de cochinita pibil"]),
    "chiles_en_nogada": ("SD", "Chiles en nogada", ["chiles en nogada", "chile en nogada plato granada"]),
    "chile_relleno": ("SP", "Chile relleno", ["chile relleno capeado", "chiles rellenos de queso"]),
    "menudo": ("SP", "Menudo", ["menudo rojo mexicano", "pancita menudo plato"]),
    "flautas": ("SP", "Flautas", ["flautas de pollo con crema", "tacos dorados mexicanos"]),
    "huaraches": ("SP", "Huaraches", ["huarache mexicano comida", "huaraches de bistec"]),
    "sopa_de_tortilla": ("SP", "Sopa de tortilla", ["sopa de tortilla mexicana", "sopa azteca"]),
    "camarones_a_la_diabla": ("SP", "Camarones a la diabla", ["camarones a la diabla", "camarones enchilados plato"]),
    "rajas_con_crema": ("SP", "Rajas con crema", ["rajas con crema y elote", "rajas poblanas con crema"]),
    "chiles_toreados": ("PS", "Chiles toreados", ["chiles toreados", "chiles jalapeños asados"]),
    "carnitas": ("S", "Carnitas", ["carnitas michoacanas", "tacos de carnitas"]),
    "barbacoa": ("S", "Barbacoa", ["barbacoa de borrego", "tacos de barbacoa"]),
    "frijoles_charros": ("S", "Frijoles charros", ["frijoles charros", "frijoles charros en olla"]),
    "molletes": ("S", "Molletes", ["molletes con pico de gallo", "molletes mexicanos"]),
    # --- Mexicana ácida ---
    "aguachile": ("SAP", "Aguachile", ["aguachile verde de camarón", "aguachile rojo"]),
    "coctel_de_camaron": ("SAP", "Cóctel de camarón", ["coctel de camarón mexicano", "campechana de mariscos"]),
    "pico_de_gallo": ("SAP", "Pico de gallo", ["pico de gallo salsa", "salsa mexicana pico de gallo en plato"]),
    "ensalada_de_nopales": ("SA", "Ensalada de nopales", ["ensalada de nopales", "nopales con jitomate y cebolla"]),
    # --- Mexicana dulce / dulce-ácido-picante ---
    "mangonada": ("DAP", "Mangonada", ["mangonada con chamoy", "chamoyada de mango"]),
    "fruta_con_chile": ("DAP", "Fruta con chile", ["fruta picada con chile y limón", "pepino jícama con chile tajín"]),
    "dulces_enchilados": ("DAP", "Dulces enchilados", ["dulces de tamarindo con chile", "gomitas enchiladas"]),
    "pan_dulce": ("D", "Pan dulce", ["conchas pan dulce mexicano", "pan dulce mexicano canasta"]),
    "flan": ("D", "Flan", ["flan napolitano", "flan mexicano con caramelo"]),
    "arroz_con_leche": ("D", "Arroz con leche", ["arroz con leche y canela", "arroz con leche mexicano"]),
    "pastel_tres_leches": ("D", "Pastel tres leches", ["pastel tres leches", "tres leches cake slice"]),
    # --- Refuerzos de amargo y ácido ---
    "cafe_negro": ("M", "Café negro", ["taza de café negro", "café americano taza", "espresso cup"]),
    "cafe_de_olla": ("DM", "Café de olla", ["café de olla en jarrito de barro"]),
    "chocolate_amargo": ("DM", "Chocolate amargo", ["dark chocolate bar pieces", "chocolate amargo tableta"]),
    "toronja": ("AM", "Toronja", ["toronja partida a la mitad", "grapefruit halves"]),
    "limones": ("A", "Limones", ["limones partidos", "limes cut in half"]),
    "ensalada_de_arugula": ("SM", "Ensalada de arúgula", ["ensalada de arúgula", "arugula salad plate"]),
}

# Búsquedas extra para clases que ya existen en Food-101 (versión mexicana)
REFUERZOS = {
    "tacos": ["tacos al pastor taquería", "tacos de bistec con salsa", "tacos mexicanos de suadero", "tacos de asada con cilantro y cebolla"],
    "chicken_quesadilla": ["quesadillas mexicanas de comal", "quesadillas de huitlacoche"],
    "nachos": ["nachos con queso y jalapeños mexicanos"],
    "guacamole": ["guacamole en molcajete"],
    "churros": ["churros con chocolate mexicanos"],
    "huevos_rancheros": ["huevos rancheros plato mexicano", "huevos divorciados"],
}


def tabla():
    """Devuelve {clase: (sabores_letras, nombre_es)} para TODAS las clases."""
    t = dict(FOOD101)
    t.update({k: (v[0], v[1]) for k, v in EXTRA.items()})
    return t


def vector_sabores(letras):
    return [int(any(LETRAS[l] == s for l in letras)) for s in SABORES]
