from enum import Enum
from typing import List


def _get_filenames(place: str):
    output = place.split(',')
    assert len(output) > 1
    output = output[0] + "_" + place.split(',')[-1]+"_recent.xml"
    output = output.replace(' ', "")
    return output


class PredefinedDatasets(Enum):
    netherlands = "netherlands"
    denmark = "denmark"
    poland = "poland"

    def __str__(self):
        return self.value
    
    def get_dataset_filenames(self) -> List[str]:
        if self.value==PredefinedDatasets.netherlands.value:
            return NETHERLANDS_RAW
        elif self.value==PredefinedDatasets.denmark.value:
            return DENMARK_RAW
        elif self.value==PredefinedDatasets.poland.value:
            return POLAND_RAW
        else:
            raise Exception()


WANDB_API_KEY = "48556bcc0c0de317a54d295e3368da0f268e8e35"
ALL = [
    "Wrocław, województwo dolnośląskie, Polska",
    "Gdańsk, województwo pomorskie, Polska",
    "Poznań, województwo wielkopolskie, Polska",
    "Warszawa, województwo mazowieckie, Polska",
    "Kraków, województwo małopolskie, Polska",
    "Berlin, Niemcy",
    "Mediolan, Lombardia, Włochy",
    "Amsterdam, Holandia Północna, Niderlandy, Holandia",
    # "Londyn, Greater London, Anglia, Wielka Brytania", # too big
    "Budapeszt, Środkowe Węgry, Węgry",
    "Sztokholm, Solna kommun, Stockholm County, Szwecja",
    "Oslo, Norwegia",
    "Wilno, Samorząd miasta Wilna, Okręg wileński, Litwa",
    "Bruksela, Brussels-Capital, Belgia",
    "Rzym, Roma Capitale, Lacjum, Włochy",
    "Florencja, Metropolitan City of Florence, Toskania, Włochy",
    "Bolonia, Emilia-Romania, Włochy",
    "Lizbona, Lisbon, Portugalia",
    "Madryt, Área metropolitana de Madrid y Corredor del Henares, Wspólnota Madrytu, Hiszpania",
    "Sewilla, Sevilla, Andaluzja, Hiszpania",
    "Walencja, Comarca de València, Walencja, Wspólnota Walencka, Hiszpania",
    "Barcelona, Barcelonès, Barcelona, Katalonia, 08001, Hiszpania",
    "Bilbao, Biscay, Kraj Basków, Hiszpania",
    "Saragossa, Zaragoza, Saragossa, Aragonia, Hiszpania",
    "Marsylia, Marseille, Bouches-du-Rhône, Prowansja-Alpy-Lazurowe Wybrzeże, Francja metropolitalna, 13000, Francja",
    "Lyon, Métropole de Lyon, Departemental constituency of Rhône, Owernia-Rodan-Alpy, Francja metropolitalna, Francja",
    "Bordeaux, Żyronda, Nowa Akwitania, Francja metropolitalna, Francja",
    "Paryż, Ile-de-France, Francja metropolitalna, Francja",
    "Rennes, Ille-et-Vilaine, Brittany, Francja metropolitalna, Francja",
    "Lille, Nord, Hauts-de-France, Francja metropolitalna, Francja ",
    "Amiens, Somme, Hauts-de-France, Francja metropolitalna, Francja",
    "Dublin, Dublin 1, Leinster, Irlandia",
    "Rotterdam, Holandia Południowa, Niderlandy, Holandia",
    "Haga, Holandia Południowa, Niderlandy, Holandia",
    "Dordrecht, Holandia Południowa, Niderlandy, Holandia",
    "Antwerpia, Flanders, Belgia",
    "Essen, Nadrenia Północna-Westfalia, Niemcy",
    "Hanower, Region Hannover, Dolna Saksonia, Niemcy",
    "Monachium, Bawaria, Niemcy",
    "Berno, Bern-Mittelland administrative district, Bernese Mittelland administrative region, Berno, Szwajcaria",
    "Zurych, District Zurich, Zurych, Szwajcaria",
    "Bazylea, Basel-City, Szwajcaria",
    "Salzburg, 5020, Austria",
    "Wiedeń, Austria",
    "Praga, Czechy",
    "Malmo, Malmö kommun, Skåne County, Szwecja",
    "Central Region, Malta",
    "Ljubljana, Upravna Enota Ljubljana, Słowenia",
    "Zagrzeb, City of Zagreb, Chorwacja",
    "Budapeszt, Środkowe Węgry, Węgry",
    "Bukareszt, Rumunia",
    "Helsinki, Helsinki sub-region, Uusimaa, Southern Finland, Mainland Finland, Finlandia",
    "Wenecja, Venezia, Wenecja Euganejska, Włochy",
    "Arnhem, Geldria, Niderlandy, Holandia",
    "Bratysława, Kraj bratysławski, Słowacja",
    "Tallinn, Prowincja Harju, Estonia",
    "Ryga, Liwonia, Łotwa",
    # "Neapol, Napoli, Kampania, Włochy",#corrupted
    "Bari, Apulia, Włochy",
    "Cardiff, Walia, CF, Wielka Brytania",
    "Birmingham, Attwood Green, West Midlands Combined Authority, Anglia, Wielka Brytania",
    "Lwów, Lviv Urban Hromada, Rejon lwowski, Obwód lwowski, Ukraina"
]
WROCLAW = ["Wrocław, województwo dolnośląskie, Polska"]
GDANSK = ["Gdańsk, województwo pomorskie, Polska"]
WALBRZYCH = ["Wałbrzych, Lower Silesian Voivodeship, Poland"]
NETHERLANDS = [
    "Amsterdam, North Holland, Netherlands",
    "Rotterdam, South Holland, Netherlands",
    "The Hague, South Holland, Netherlands",
    "Utrecht, Netherlands",
    "Eindhoven, North Brabant, Netherlands",
    "Groningen, Netherlands",
    "Tilburg, North Brabant, Netherlands",
    "Almere, Flevoland, Netherlands",
    "Breda, North Brabant, Netherlands",
    "Nijmegen, Gelderland, Netherlands"
]
NETHERLANDS_RAW = [
    _get_filenames(place) for place in NETHERLANDS
]
DENMARK = [
    "Antwerpia, Flanders, Belgia",
    "Ghent, Gent, Flandria Wschodnia, Flanders, Belgia",
    "Charleroi, Hainaut, Walonia, Belgia",
    "Liège, Walonia, 4000, Belgia",
    "Bruksela, Brussels, Brussels-Capital, Belgia",
    "Schaerbeek - Schaarbeek, Brussels-Capital, Belgia",
    "Anderlecht, Brussels-Capital, 1070, Belgia",
    "Brugia, Flandria Zachodnia, Flanders, Belgia",
    "Namur, Walonia, Belgia"
]
DENMARK_RAW = [
    _get_filenames(place) for place in DENMARK
]
POLAND = [

]
POLAND_RAW = [
    _get_filenames(place) for place in POLAND
]


TRAINING_SET = [
                "Copenhagen Municipality, Region Stołeczny, Dania",
                "Gmina Aarhus, Jutlandia Środkowa, Dania",
                "Odense Kommune, Dania Południowa, Dania",
                "Gmina Aalborg, Jutlandia Północna, Dania",
                "Frederiksberg Municipality, Region Stołeczny, Dania",
                "Gimina Gentofte, Region Stołeczny, Dania",
                "Warszawa, województwo mazowieckie, Polska",
                "Łódź, województwo łódzkie, Polska",
                "Kraków, województwo małopolskie, Polska",
                "Poznań, powiat poznański, województwo wielkopolskie, Polska",
                "Szczecin, województwo zachodniopomorskie, Polska",
                "Bydgoszcz, województwo kujawsko-pomorskie, Polska",
                "Lublin, województwo lubelskie, Polska",
                "Białystok, powiat białostocki, województwo podlaskie, Polska",
                "Gdynia, województwo pomorskie, Polska",
                "Katowice, Górnośląsko-Zagłębiowska Metropolia, województwo śląskie, Polska",
                "Antwerpia, Flanders, Belgia",
                "Ghent, Gent, Flandria Wschodnia, Flanders, Belgia",
                "Charleroi, Hainaut, Walonia, Belgia",
                "Liège, Walonia, 4000, Belgia",
                "Bruksela, Brussels, Brussels-Capital, Belgia",
                "Schaerbeek - Schaarbeek, Brussels-Capital, Belgia",
                "Anderlecht, Brussels-Capital, 1070, Belgia",
                "Brugia, Flandria Zachodnia, Flanders, Belgia",
                "Namur, Walonia, Belgia",
                "Utrecht, Niderlandy, Holandia",
                "Birmingham, West Midlands Combined Authority, Anglia, Wielka Brytania",
                "Lyon, Métropole de Lyon, Rhône, Owernia-Rodan-Alpy, Francja metropolitalna, Francja",
                "Tuluza, Górna Garonna, Oksytania, Francja metropolitalna, Francja",
                "Brno, okres Brno-město, Kraj południowomorawski, Południowo-wschodni, Czechy",
                "Ostrawa, Powiat Ostrawa-miasto, Kraj morawsko-śląski, Czechy",
                "Bratysława, Kraj bratysławski, Słowacja",
                "Oslo, Norwegia",
                "Stockholms kommun, Stockholm County, Szwecja",
                "Malmo, Malmö kommun, Skåne County, Szwecja",
                "Helsinki, Helsinki sub-region, Uusimaa, Southern Finland, Mainland Finland, Finlandia",
                "Ryga, Liwonia, Łotwa",
                "Berno, Bern-Mittelland administrative district, Bernese Mittelland administrative region, Berno, Szwajcaria",
                "Zurych, District Zurich, Zurych, Szwajcaria",
                "Brema, Free Hanseatic City of Bremen, Niemcy",
                "Hanower, Region Hannover, Dolna Saksonia, Niemcy",
                "Essen, Nadrenia Północna-Westfalia, Niemcy",
                "Lipsk, Saksonia, Niemcy",
                "Mediolan, Lombardia, Włochy",
                "Florencja, Metropolitan City of Florence, Toskania, Włochy",
                "Sewilla, Sevilla, Andaluzja, Hiszpania",
                "Saragossa, Zaragoza, Saragossa, Aragonia, Hiszpania",
                "Walencja, Comarca de València, Walencja, Wspólnota Walencka, Hiszpania",
                "Porto, Portugalia",
                "Split, Grad Split, Split-Dalmatia County, Chorwacja"
                ]

TRAINING_SET_RAW = [
    _get_filenames(place) for place in TRAINING_SET
]

VALIDATION_SET = [
    "Wrocław, województwo dolnośląskie, Polska",
    "Gdańsk, województwo pomorskie, Polska",
    "Monachium, Bawaria, Niemcy",
    "Bolonia, Emilia-Romania, Włochy",
    "Eindhoven, Brabancja Północna, Niderlandy, Holandia"
]

VALIDATION_SET_RAW = [
    _get_filenames(place) for place in VALIDATION_SET
]