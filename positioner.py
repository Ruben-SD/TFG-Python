class Positioner:
    def __init__(self, config):
        position_config = config['smartphone']['position'] 
        self.initial_position = position_config['x'], position_config['y']
        self.position = self.initial_position

    def set_position(self, position):
        self.position = tuple([sum(x) for x in zip(self.initial_position, position)])

    def move_by(self, amount):
        self.position = self.position[0] + amount[0], self.position[1] + amount[1]

    def update_position(self, dt):
        pass