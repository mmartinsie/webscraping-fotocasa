class Home:
    # Attributes
    price = district = rooms = baths = size = floor = url = type = parking = None

    # Constructor
    def __init__(self, url, district):
        self.district = district
        self.url = url

    def toString(self):
        print('Precio:', self.price)
        print('Distrito', self.district)
        print('Tipo de inmueble:', self.type)
        print('Nº Habitaciones:', self.rooms)
        print('Nº Baños:', self.baths)
        print('Tamaño:', self.size)
        print('Planta:', self.floor)
        print('Parking:', self.parking)
        print('Link: ', self.url)

        
