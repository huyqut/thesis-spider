from geograpy.extraction import Extractor
from geograpy.places import PlaceContext


def get_place_context(url=None, text=None):
    e = Extractor(url=url, text=text)
    e.find_entities()

    pc = PlaceContext(e.places, e.people, e.organs)
    pc.set_countries()
    pc.set_regions()
    pc.set_cities()
    pc.set_other()

    return pc



# url = 'http://www.bbc.com/news/world-us-canada-39821789'
# places = get_place_context(url=url)
# len(places)