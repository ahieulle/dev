from bs4 import BeautifulSoup
import requests

ligue1 = "https://www.betclic.fr/football/ligue-1-e4"
parser = "html.parser"

soup = BeautifulSoup(requests.get(ligue1).content, parser)

match_odds = soup.select("[class~=match-odds]")

for odds in match_odds:
    cotes1x2 = odds.select(".odd-button")
    cotes = []
    for cote in cotes1x2:
        cotes.append(float(cote.text.replace(",",".")))
    match = odds.parent["data-track-event-name"]
    date = odds.parent.parent.parent.time["datetime"]
    print(match, cotes, date)
