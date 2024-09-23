new_tests_inputs = [
    #Test Case 1
    {
    "optimal": False,
    "infinite": False,
    "map": [
        ['S', 'I', 'S', 'S', 'S', 'I', 'S', 'S', 'I', 'S'],
        ['B', 'S', 'S', 'I', 'S', 'S', 'S', 'I', 'S', 'S'],
        ['S', 'S', 'I', 'S', 'S', 'I', 'S', 'S', 'S', 'S'],
        ['I', 'S', 'S', 'S', 'I', 'S', 'S', 'I', 'S', 'S'],
        ['S', 'I', 'S', 'S', 'S', 'S', 'I', 'S', 'S', 'I'],
        ['S', 'S', 'I', 'S', 'S', 'S', 'S', 'I', 'S', 'S'],
    ],
    "pirate_ships": {
        'pirate_ship_1': {"location": (1, 0), "capacity": 2},
        'pirate_ship_2': {"location": (1, 0), "capacity": 2}
    },
    "treasures": {
        'treasure_1': {"location": (0, 1), "possible_locations": ((0, 1), (3, 0), (4, 9)), "prob_change_location": 0.2},
        'treasure_2': {"location": (5, 7), "possible_locations": ((5, 7), (3, 7), (0, 8)), "prob_change_location": 0.1}
    },
    "marine_ships": {
        'marine_1': {"index": 0, "path": [(1, 1), (2, 1), (3, 1)]},
        'marine_2': {"index": 2, "path": [(4, 1), (4, 2), (4, 3), (4, 4)]}
    },
    "turns to go": 100
    },
#Test Case 2: Basic Scenario
{
    "optimal": False,
    "infinite": False,
    "map": [
        ['S', 'I', 'S', 'S'],
        ['B', 'S', 'I', 'S'],
        ['S', 'S', 'S', 'S'],
        ['I', 'S', 'S', 'I']
    ],
    "pirate_ships": {
        'pirate_ship_1': {"location": (1, 0), "capacity": 2}
    },
    "treasures": {
        'treasure_1': {"location": (0, 1), "possible_locations": ((0, 1), (3, 0)), "prob_change_location": 0.3}
    },
    "marine_ships": {
        'marine_1': {"index": 0, "path": [(2, 1), (2, 2)]}
    },
    "turns to go": 20
},
#Test Case 3: Multiple Pirates and Marines- לא רץ
{
    "optimal": False,
    "infinite": False,
    "map": [
        ['S', 'I', 'S', 'S', 'I', 'S'],
        ['B', 'S', 'S', 'I', 'S', 'S'],
        ['S', 'S', 'I', 'S', 'S', 'I'],
        ['S', 'I', 'S', 'S', 'I', 'S'],
        ['I', 'S', 'S', 'I', 'S', 'S'],
        ['B', 'S', 'I', 'S', 'S', 'I']
    ],
    "pirate_ships": {
        'pirate_ship_1': {"location": (1, 0), "capacity": 2},
        'pirate_ship_2': {"location": (1, 0), "capacity": 2}
    },
    "treasures": {
        'treasure_1': {"location": (0, 1), "possible_locations": ((0, 1), (4, 3), (4, 0)), "prob_change_location": 0.2},
        'treasure_2': {"location": (2, 5), "possible_locations": ((2, 5), (5, 5), (1, 3)), "prob_change_location": 0.25}
    },
    "marine_ships": {
        'marine_1': {"index": 0, "path": [(1, 2), (0, 2), (0, 3)]},
        'marine_2': {"index": 1, "path": [(4, 5), (4, 4), (4, 3)]}
    },
    "turns to go": 100
},

#Test Case 4: High Complexity with Many Elements- לא רץ
{
    "optimal": False,
    "infinite": False,
    "map": [
        ['S', 'I', 'S', 'S', 'I', 'S', 'S', 'I', 'S'],
        ['B', 'S', 'S', 'I', 'S', 'S', 'I', 'S', 'S'],
        ['S', 'S', 'I', 'S', 'S', 'I', 'S', 'S', 'I'],
        ['S', 'I', 'S', 'S', 'I', 'S', 'S', 'I', 'S'],
        ['S', 'S', 'S', 'I', 'S', 'S', 'I', 'S', 'S'],
        ['I', 'S', 'S', 'S', 'I', 'S', 'S', 'I', 'S'],
        ['S', 'I', 'S', 'S', 'S', 'I', 'S', 'S', 'I']
    ],
    "pirate_ships": {
        'pirate_ship_1': {"location": (1, 0), "capacity": 2},
        'pirate_ship_2': {"location": (1, 0), "capacity": 2},
        'pirate_ship_3': {"location": (1, 0), "capacity": 2}
    },
    "treasures": {
        'treasure_1': {"location": (0, 1), "possible_locations": ((0, 1), (6, 8)), "prob_change_location": 0.2},
        'treasure_2': {"location": (2, 5), "possible_locations": ((2, 5), (5, 7), (3, 7)), "prob_change_location": 0.3},
        'treasure_3': {"location": (6, 8), "possible_locations": ((6, 8), (4, 6), (1, 3)), "prob_change_location": 0.4}
    },
    "marine_ships": {
        'marine_1': {"index": 0, "path": [(2, 1), (1, 1), (1, 2),(0, 2)]},
        'marine_2': {"index": 1, "path": [(5, 2), (5, 3)]},
        'marine_3': {"index": 0, "path": [(6, 3), (6, 4)]}
    },
    "turns to go": 100
},
#Test Case 5: Simple Island Treasure- רץ
{
    "optimal": False,
    "infinite": False,
    "map": [
        ['S', 'S', 'I', 'S'],
        ['B', 'S', 'S', 'S'],
        ['S', 'I', 'S', 'S'],
        ['S', 'S', 'S', 'I']
    ],
    "pirate_ships": {
        'pirate_jim': {"location": (1, 0), "capacity": 2}
    },
    "treasures": {
        'isolated_treasure': {"location": (0, 2), "possible_locations": ((0, 2), (2, 1), (3, 3)), "prob_change_location": 0.3}
    },
    "marine_ships": {
        'marine_scout': {"index": 0, "path": [(1, 1), (1, 2)]}
    },
    "turns to go": 100
},
#Test Case 6: High Density Scenario with Stochastic Complexity- לא רץ לא  יודע למה
{
    "optimal": False,
    "infinite": False,
    "map": [
        ['I', 'S', 'S', 'I', 'S', 'I', 'S', 'S'],
        ['S', 'S', 'I', 'S', 'I', 'S', 'S', 'I'],
        ['S', 'I', 'S', 'I', 'S', 'I', 'S', 'S'],
        ['B', 'S', 'S', 'S', 'I', 'S', 'S', 'I'],
        ['I', 'S', 'I', 'S', 'S', 'I', 'S', 'S'],
        ['S', 'I', 'S', 'S', 'S', 'I', 'S', 'I'],
        ['S', 'S', 'I', 'S', 'I', 'S', 'S', 'S'],
        ['I', 'S', 'S', 'I', 'S', 'S', 'I', 'S']
    ],
    "pirate_ships": {
        'queen_anne': {"location": (3, 0), "capacity": 2},
        'davy_jones': {"location": (3, 0), "capacity": 2},
        'flying_dutchman': {"location": (3, 0), "capacity": 2}
    },
    "treasures": {
        'emerald_ring': {"location": (0, 0), "possible_locations": ((0, 0), (4, 2), (2, 5)), "prob_change_location": 0.15},
        'silver_chalice': {"location": (1, 7), "possible_locations": ((1, 7), (3, 7), (2, 5)), "prob_change_location": 0.25},
        'bronze_dagger': {"location": (3, 7), "possible_locations": ((3, 7), (2, 5), (0, 0)), "prob_change_location": 0.2}
    },
    "marine_ships": {
        'naval_fleet': {"index": 2, "path": [(0, 2), (0, 1), (1, 1),(1, 0), (2, 0), (3, 0),(3, 1), (3, 2), (3, 3)]}
    },
    "turns to go": 100
},
#Test 7: Island with Inaccessible Treasure רץ
{
    "optimal": False,
    "infinite": False,
    "map": [
        ['I', 'I', 'I', 'I'],
        ['I', 'B', 'I', 'I'],
        ['I', 'I', 'I', 'I'],
        ['I', 'I', 'I', 'I']
    ],
    "pirate_ships": {
        'lonely_pirate': {"location": (1, 1), "capacity": 2}
    },
    "treasures": {
        'hidden_gem': {"location": (2, 2), "possible_locations": ((2, 2),), "prob_change_location": 0.0}
    },
    "marine_ships": {},
    "turns to go": 100
},
#Test 8: Marines Blocking Access to an Island רץ
{
    "optimal": False,
    "infinite": False,
    "map": [
        ['S', 'S', 'S', 'S', 'S'],
        ['S', 'I', 'S', 'I', 'S'],
        ['S', 'S', 'I', 'S', 'S'],
        ['B', 'S', 'S', 'S', 'S']
    ],
    "pirate_ships": {
        'blocked_pirate': {"location": (3, 0), "capacity": 7}
    },
    "treasures": {
        'isolated_treasure': {"location": (2, 2), "possible_locations": ((2, 2),), "prob_change_location": 0.0}
    },
    "marine_ships": {
        'marine_blockade': {"index": 0, "path": [(2, 1), (2, 3), (3, 2), (3, 3), (2, 3)]},
        'omri': {"index": 1, "path": [(2, 1), (2, 3), (3, 2), (3, 3), (2, 3)]},
        'omri2': {"index": 2, "path": [(2, 1), (2, 3), (3, 2), (3, 3), (2, 3)]},
        'omriai': {"index": 3, "path": [(2, 1), (2, 3), (3, 2), (3, 3), (2, 3)]},
        'omri22': {"index": 4, "path": [(2, 1), (2, 3), (3, 2), (3, 3), (2, 3)]}
    },
    "turns to go": 100
},
    #Test Case 9: pirates- לא עובד על יותר מ70 תורות
    {
        "optimal": False,
        "infinite": False,
        "map": [['S', 'S', 'I', 'S', 'S', 'S', 'S', 'S', 'S', 'S'],
                ['S', 'S', 'I', 'S', 'S', 'S', 'S', 'S', 'S', 'S'],
                ['B', 'S', 'I', 'S', 'S', 'S', 'S', 'S', 'S', 'S'],
                ['S', 'S', 'I', 'S', 'S', 'S', 'S', 'S', 'I', 'S'],
                ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S'],
                ['S', 'S', 'S', 'S', 'I', 'S', 'S', 'S', 'S', 'I']],

        "pirate_ships": {'pirate_ship_1': {"location": (2, 0),
                                           "capacity": 2},

                         },
        "treasures": {'treasure_1': {"location": (0, 2),
                                     "possible_locations": ((0, 2), (1, 2), (3, 2)),
                                     "prob_change_location": 0.2},
                      'treasure_2': {"location": (2, 2),
                                     "possible_locations": ((0, 2), (2, 2), (3, 2)),
                                     "prob_change_location": 0.1},
                      'treasure_3': {"location": (3, 8),
                                     "possible_locations": ((3, 8), (3, 2), (5, 4)),
                                     "prob_change_location": 0.3},
                      'magical treasure': {"location": (5, 9),
                                           "possible_locations": ((5, 9), (5, 4)),
                                           "prob_change_location": 0.4}
                      },
        "marine_ships": {'marine_1': {"index": 0,
                                      "path": [(1, 1), (2, 1)]},
                         "larry the marine": {"index": 0,
                                              "path": [(5, 6), (4, 6), (4, 7)]},
                         },
        "turns to go": 100
    },
]


"""{
        "optimal": False,
        "infinite": False,
        "map": [['S', 'S', 'I', 'S', 'S', 'S', 'S', 'S', 'S', 'S'],
                ['S', 'S', 'I', 'S', 'S', 'S', 'S', 'S', 'S', 'S'],
                ['B', 'S', 'I', 'S', 'S', 'S', 'S', 'S', 'S', 'S'],
                ['S', 'S', 'I', 'S', 'S', 'S', 'S', 'S', 'I', 'S'],
                ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S'],
                ['S', 'S', 'S', 'S', 'I', 'S', 'S', 'S', 'S', 'I']],

        "pirate_ships": {'pirate_ship_1': {"location": (2, 0),
                                           "capacity": 2},
                         'pirate_bob': {"location": (2, 0),
                                        "capacity": 2},
                         'bob the pirate': {"location": (2, 0),
                                            "capacity": 2}
                         },
        "treasures": {'treasure_1': {"location": (0, 2),
                                     "possible_locations": ((0, 2), (1, 2), (3, 2)),
                                     "prob_change_location": 0.2},
                      'treasure_2': {"location": (2, 2),
                                     "possible_locations": ((0, 2), (2, 2), (3, 2)),
                                     "prob_change_location": 0.1},
                      'treasure_3': {"location": (3, 8),
                                     "possible_locations": ((3, 8), (3, 2), (5, 4)),
                                     "prob_change_location": 0.3},
                      'magical treasure': {"location": (5, 9),
                                           "possible_locations": ((5, 9), (5, 4)),
                                           "prob_change_location": 0.4}
                      },
        "marine_ships": {'marine_1': {"index": 0,
                                      "path": [(1, 1), (2, 1)]},
                         "larry the marine": {"index": 0,
                                              "path": [(5, 6), (4, 6), (4, 7)]},
                        'marine_2': {"index": 1,
                                      "path": [(1, 1), (2, 1)]},
                         "larry the ": {"index": 1,
                                              "path": [(5, 6), (4, 6), (4, 7)]},
                        'marine_3': {"index": 1,
                                      "path": [(1, 1), (2, 1)]},
                         "larry the 23rewqfq": {"index": 2,
                                              "path": [(5, 6), (4, 6), (4, 7)]},
                         },
        "turns to go": 75
    }"""