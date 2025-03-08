#PROMPT of generating concept
PROMPT_1 = """

Definition:
A general concept is a broad idea or category that captures common attributes or qualities shared by multiple specific instances or objects, which may be concrete or abstract. It simplifies complex information by grouping similar instances together. For example: 1) The concept of “animal” encompasses all living organisms that can move and consume food. 2) The concept of “freedom” describes a state of being free from constraints or oppression, applicable in various social, political, and personal contexts.

Instruction:
Given a set of captions describing one single image, determine the first-level main general concept, so avoid being too general. The first-level concept should describe a general idea of the original caption. The concept should be a short phrase or a brief phrase that captures the general idea of the image (represented by the captions). Emphasize more on events or themes rather than objects or actions. Concepts MUST use canonical vocabulary (e.g. singular; present tense). Do not provide explanations. Write the concept in an <output> section.

Example 1:
Captions:
The tray on the bed has a pastry and two mugs on it.
A tray with coffee and a pastry on it.
A tray is full of breakfast foods and drinks on a bed.
coffee cream and a croissant on a tray
A tray on a bed with food and drink.
Output:
<output>Breakfast</output>

Example 2:
Captions:
Man in dress shirt and orange tie standing inside a building.
a male with a beard and orange tie
A man wearing a neck tie and a white shirt.
A man posing for the picture in a building
A man dressed in a shirt and tie standing in a lobby.
Output:
<output>Business Attire</output>

Example 3:
Captions:
Many small children are posing together in the black and white photo.
A vintage school picture of grade school aged children.
A black and white photo of a group of kids.
A group of children standing next to each other.
A group of children standing and sitting beside each other.
Output:
<output>Education</output>

Example 4:
Captions:
there is a very tall giraffe standing in the wild
A giraffe is standing by some brush in a field.
A giraffe standing in a dirt field near a tree.
A single giraffe standing in a brushy area looking at the photographer
A giraffe standing in dry dead brush on the savannah.
Output:
<output>Wildlife</output>

Example 5:
Captions:
Ornate wedding cake ready at the hotel reception
A tall multi layer cake sitting on top of a blue table cloth.
A wedding cake with flowers in a banquet hall.
Nicely decorated three tier wedding cake with topper.
A very ornate, three layered wedding cake in a banquet room.
Output:
<output>Wedding</output>

Example 6:
Captions:
A group of red, yellow and orange fruit mixed together.
a bunch of yellow and orange fruits in a pile
A bunch of yellow and orange fruit in varied sizes.
Pears and oranges have dirt and brown spots.
A picture of some fruit on a table.
Output:
<output>Food</output>

Example 7:
Captions:
An adult standing behind a little girl while holding an umbrella.
The two people are standing under an umbrella in the rain.
Two girls under a large umbrella in the rain
A couple of people standing with a umbrella.
A child and person stand under an umbrella.
Output:
<output>Parenting</output>

Example 8:
Captions:
Woman snow boarding off of a cliff in the air.
A person doing a trick on a snowboard over a hill.
A snowboarder hitting a trick on a mountain.
a snowboarder in a black jacket is doing a trick
a person on a snow board flying in the air down the snow
Output:
<output>Sport</output>

Example 9:
Captions:
a person putting some pastries into a bag
A baker is placing their goods inside a bag.
A person working at a store selling pasties.
A person putting doughnuts into a bag in a shop.
Pans of potatoes and a person wearing a red apron in a commercial kitchen.
Output:
<output>Commerce</output>

Example 10:
Captions:
A laptop with a stationary mouse attached to it.
A laptop computer with an attached trackball device, showing the Windows logo on the screen.
The new Sony Y series laptop is connected to a special mouse.
A Windows laptop computer with a large mouse.
A Windows laptop with a corded mouse on top of a counter.
Output:
<output>Technology</output>

Example 11:
Captions:
A giraffe is peeking around the side of a wall at the camera.
A camel or a giraffe is playing with the camera-man.
A giraffe has its head pressed against the wall.
A half of face of a giraffe and a tree.
A giraffe standing next to a building near a tree.
Output:
<output>Wildlife</output>

Example 12:
Captions:
A walk in shower next to a toilet with a wooden seat.
The neat bathroom has green trim on the tile.
A toilet with a wood seat and a tiled floor.
This bathroom has a pattern of blue tiles on the floor.
A bathroom with blue tile outline around the floor.
Output:
<output>Home Design</output>

Example 13:
Captions:
A pair of scissors sitting on top of a table.
A pair of scissors, tape, and wrapping paper lie on a wooden surface.
Some scrap book scissors are on a brown table.
a pair of scissor that is laying on a wooden table.
A pair of scissors and fabric on a wood table.
Output:
<output>Crafting</output>

Example 14:
Captions:
A fighter jet plane ascending into the sky.
A fighter jet gaining altitude in a cloudy sky.
An aircraft is flying in a cloudy sky.
A military jet taking off into the sky.
A nice jet plane flying into some grey skies.
Output:
<output>Defense</output>

Example 15:
Captions:
A man is in a kitchen making pizzas.
Man in apron standing on front of oven with pans and bakeware.
A baker is working in the kitchen rolling dough.
A person standing by a stove in a kitchen.
A table with pies being made and a person standing near a wall with pots and pans hanging on the wall.
Output:
<output>Food Preparation</output>

Example 16:
Captions:
A parking meter next to a handicap parking space.
A parking meter that is placed right next to a parking lot.
a parking meter on the side of a road with a red light showing.
The parking meter in the parking lot has seven fifteen on it.
A parking meter sitting next to a parking space.
Output:
<output>Urban Regulation</output>

Example 17:
Captions:
THERE IS A GROUP OF PEOPLE SITTING AT A TABLE.
a group of people sitting at a long dining table in a restaurant.
A bunch of people sitting at a table having a discussion.
A large group of people at a table.
A group of people sitting around a table having a meal.
Output:
<output>Social Gathering</output>

Example 18:
Captions:
A young boy holding Nintendo Wii game controllers.
A little boy is staring attentively while playing a video game.
A small child holds two video game controllers.
A boy in grey shirt holding a Nintendo Wii controller.
a close up of a young child playing Nintendo Wii.
Output:
<output>Technology</output>

Example 19:
Captions:
View of a city skyline and train yard at dusk.
A city with skyscrapers, other buildings, and trains.
In a city, the buildings are illuminated while trains sit below.
An urban skyline with a train-yard in the foreground.
A city scape scene with a train yard in the foreground.
Output:
<output>Urban Nightlife</output>

Example 20:
Captions:
a group of young people playing baseball in a field.
Baseball players are on the field during a game.
A young baseball player tagging up at third base.
five baseball players with a runner on the base.
Baseball players in their uniforms on a baseball field.
Output:
<output>Sport</output>

Example 21:
Captions:
Someone is driving a cart near a single engine airplane.
a small air plane on a small run way.
A small airplane that is parked near a run way.
A small airplane sitting next to an airport runway.
A crop dusting plane next to grassy field and tree.
Output:
<output>Farming</output>

Example 22:
Captions:
There is a map in the street of the city.
a bus stop map in a city near a water fountain.
The subway stop Square Victoria entrance and the map of the neighborhood.
A map and street sign with building in background.
A map of the town in the middle of the street with buildings in the background.
Output:
<output>Public Transport</output>

Example 23:
Captions:
a green vase with some flowers and greenery.
A vase sitting next to a mirror filled with flowers.
This is a still life, slightly blurry, with a tea kettle and a floral arrangement.
A vase with roses and carnations in it by a mirror.
Flowers sit in a vase beside a tea pot.
Output:
<output>Domestic Scene</output>

Example 24:
Captions:
A utility truck is parked in the street beside traffic cones.
A street with orange cones and a work truck on it.
Traffic cones at the entrance of a lot where there is construction.
A construction truck is in front of the huge building.
a bunch of orange cones sitting in the road.
Output:
<output>Construction</output>

"""
PROMPT_2 = """
Example 1:
Captions:
The tray on the bed has a pastry and two mugs on it.
A tray with coffee and a pastry on it.
A tray is full of breakfast foods and drinks on a bed.
coffee cream and a croissant on a tray
A tray on a bed with food and drink.
Level1 Concept: Breakfast
Output:
<output>Meal</output>

Example 2:
Captions:
Man in dress shirt and orange tie standing inside a building.
a male with a beard and orange tie
A man wearing a neck tie and a white shirt.
A man posing for the picture in a building
A man dressed in a shirt and tie standing in a lobby.
Level1 Concept: Business Attire
Output:
<output>Business</output>

Example 3:
Captions:
Many small children are posing together in the black and white photo.
A vintage school picture of grade school aged children.
A black and white photo of a group of kids.
A group of children standing next to each other.
A group of children standing and sitting beside each other.
Level1 Concept: Education
Output:
<output>Learning</output>

Example 4:
Captions:
there is a very tall giraffe standing in the wild
A giraffe is standing by some brush in a field.
A giraffe standing in a dirt field near a tree.
A single giraffe standing in a brushy area looking at the photographer
A giraffe standing in dry dead brush on the savannah.
Level1 Concept: Wildlife
Output:
<output>Nature</output>

Example 5:
Captions:
Ornate wedding cake ready at the hotel reception
A tall multi layer cake sitting on top of a blue table cloth.
A wedding cake with flowers in a banquet hall.
Nicely decorated three tier wedding cake with topper.
A very ornate, three layered wedding cake in a banquet room.
Level1 Concept: Wedding
Output:
<output>Celebration</output>

Example 6:
Captions:
A group of red, yellow and orange fruit mixed together.
a bunch of yellow and orange fruits in a pile
A bunch of yellow and orange fruit in varied sizes.
Pears and oranges have dirt and brown spots.
A picture of some fruit on a table.
Level1 Concept: Food
Output:
<output>Nutrition</output>

Example 7:
Captions:
An adult standing behind a little girl while holding an umbrella.
The two people are standing under an umbrella in the rain.
Two girls under a large umbrella in the rain
A couple of people standing with a umbrella.
A child and person stand under an umbrella.
Level1 Concept: Parenting
Output:
<output>Family</output>

Example 8:
Captions:
Woman snow boarding off of a cliff in the air.
A person doing a trick on a snowboard over a hill.
A snowboarder hitting a trick on a mountain.
a snowboarder in a black jacket is doing a trick
a person on a snow board flying in the air down the snow
Level1 Concept: Sport
Output:
<output>Adventure</output>

Example 9:
Captions:
a person putting some pastries into a bag
A baker is placing their goods inside a bag.
A person working at a store selling pasties.
A person putting doughnuts into a bag in a shop.
Pans of potatoes and a person wearing a red apron in a commercial kitchen.
Level1 Concept: Commerce
Output:
<output>Employment</output>

Example 10:
Captions:
A laptop with a stationary mouse attached to it.
A laptop computer with an attached trackball device, showing the Windows logo on the screen.
The new Sony Y series laptop is connected to a special mouse.
A Windows laptop computer with a large mouse.
A Windows laptop with a corded mouse on top of a counter.
Level1 Concept: Technology
Output:
<output>Innovation</output>

Example 11:
Captions:
A giraffe is peeking around the side of a wall at the camera.
A camel or a giraffe is playing with the camera-man.
A giraffe has its head pressed against the wall.
A half of face of a giraffe and a tree.
A giraffe standing next to a building near a tree.
Level1 Concept: Wildlife
Output:
<output>Nature</output>

Example 12:
Captions:
A walk in shower next to a toilet with a wooden seat.
The neat bathroom has green trim on the tile.
A toilet with a wood seat and a tiled floor.
This bathroom has a pattern of blue tiles on the floor.
A bathroom with blue tile outline around the floor.
Level1 Concept: Home Design
Output:
<output>Aesthetics</output>

Example 13:
Captions:
A pair of scissors sitting on top of a table.
A pair of scissors, tape, and wrapping paper lie on a wooden surface.
Some scrap book scissors are on a brown table.
a pair of scissor that is laying on a wooden table.
A pair of scissors and fabric on a wood table.
Level1 Concept: Crafting
Output:
<output>Creativity</output>

Example 14:
Captions:
A fighter jet plane ascending into the sky.
A fighter jet gaining altitude in a cloudy sky.
An aircraft is flying in a cloudy sky.
A military jet taking off into the sky.
A nice jet plane flying into some grey skies.
Level1 Concept: Defense
Output:
<output>Protection</output>

Example 15:
Captions:
A man is in a kitchen making pizzas.
Man in apron standing on front of oven with pans and bakeware.
A baker is working in the kitchen rolling dough.
A person standing by a stove in a kitchen.
A table with pies being made and a person standing near a wall with pots and pans hanging on the wall.
Level1 Concept: Food Preparation
Output:
<output>Cooking</output>

Example 16:
Captions:
A parking meter next to a handicap parking space.
A parking meter that is placed right next to a parking lot.
a parking meter on the side of a road with a red light showing.
The parking meter in the parking lot has seven fifteen on it.
A parking meter sitting next to a parking space.
Level1 Concept: Urban Regulation
Output:
<output>Infrastructure</output>

Example 17:
Captions:
THERE IS A GROUP OF PEOPLE SITTING AT A TABLE.
a group of people sitting at a long dining table in a restaurant.
A bunch of people sitting at a table having a discussion.
A large group of people at a table.
A group of people sitting around a table having a meal.
Level1 Concept: Social Gathering
Output:
<output>Community</output>

Example 18:
Captions:
A young boy holding Nintendo Wii game controllers.
A little boy is staring attentively while playing a video game.
A small child holds two video game controllers.
A boy in grey shirt holding a Nintendo Wii controller.
a close up of a young child playing Nintendo Wii.
Level1 Concept: Technology
Output:
<output>Digital Interaction</output>

Example 19:
Captions:
View of a city skyline and train yard at dusk.
A city with skyscrapers, other buildings, and trains.
In a city, the buildings are illuminated while trains sit below.
An urban skyline with a train-yard in the foreground.
A city scape scene with a train yard in the foreground.
Level1 Concept: Urban Nightlife
Output:
<output>Infrastructure</output>

Example 20:
Captions:
a group of young people playing baseball in a field.
Baseball players are on the field during a game.
A young baseball player tagging up at third base.
five baseball players with a runner on the base.
Baseball players in their uniforms on a baseball field.
Level1 Concept: Sport
Output:
<output>Competition</output>

Example 21:
Captions:
Someone is driving a cart near a single engine airplane.
a small air plane on a small run way.
A small airplane that is parked near a run way.
A small airplane sitting next to an airport runway.
A crop dusting plane next to grassy field and tree.
Level1 Concept: Farming
Output:
<output>Agriculture</output>

Example 22:
Captions:
There is a map in the street of the city.
a bus stop map in a city near a water fountain.
The subway stop Square Victoria entrance and the map of the neighborhood.
A map and street sign with building in background.
A map of the town in the middle of the street with buildings in the background.
Level1 Concept: Public Transport
Output:
<output>Navigation</output>

Example 23:
Captions:
a green vase with some flowers and greenery.
A vase sitting next to a mirror filled with flowers.
This is a still life, slightly blurry, with a tea kettle and a floral arrangement.
A vase with roses and carnations in it by a mirror.
Flowers sit in a vase beside a tea pot.
Level1 Concept: Domestic Scene
Output:
<output>Aesthetics</output>

Example 24:
Captions:
A utility truck is parked in the street beside traffic cones.
A street with orange cones and a work truck on it.
Traffic cones at the entrance of a lot where there is construction.
A construction truck is in front of the huge building.
a bunch of orange cones sitting in the road.
Level1 Concept: Construction
Output:
<output>Urban Development</output>
"""
PROMPT_3 = """Example 1:
Captions:
The tray on the bed has a pastry and two mugs on it.
A tray with coffee and a pastry on it.
A tray is full of breakfast foods and drinks on a bed.
coffee cream and a croissant on a tray
A tray on a bed with food and drink.
Level1 Concept: Breakfast
Level2 Concept: Meal
Output:
<output>Food</output>

Example 2:
Captions:
Man in dress shirt and orange tie standing inside a building.
a male with a beard and orange tie
A man wearing a neck tie and a white shirt.
A man posing for the picture in a building
A man dressed in a shirt and tie standing in a lobby.
Level1 Concept: Business Attire
Level2 Concept: Business
Output:
<output>Career</output>

Example 3:
Captions:
Many small children are posing together in the black and white photo.
A vintage school picture of grade school aged children.
A black and white photo of a group of kids.
A group of children standing next to each other.
A group of children standing and sitting beside each other.
Level1 Concept: Education
Level2 Concept: Learning
Output:
<output>Personal Development</output>

Example 4:
Captions:
there is a very tall giraffe standing in the wild
A giraffe is standing by some brush in a field.
A giraffe standing in a dirt field near a tree.
A single giraffe standing in a brushy area looking at the photographer
A giraffe standing in dry dead brush on the savannah.
Level1 Concept: Wildlife
Level2 Concept: Nature
Output:
<output>Life</output>

Example 5:
Captions:
Ornate wedding cake ready at the hotel reception
A tall multi layer cake sitting on top of a blue table cloth.
A wedding cake with flowers in a banquet hall.
Nicely decorated three tier wedding cake with topper.
A very ornate, three layered wedding cake in a banquet room.
Level1 Concept: Wedding
Level2 Concept: Celebration
Output:
<output>Tradition</output>

Example 6:
Captions:
A group of red, yellow and orange fruit mixed together.
a bunch of yellow and orange fruits in a pile
A bunch of yellow and orange fruit in varied sizes.
Pears and oranges have dirt and brown spots.
A picture of some fruit on a table.
Level1 Concept: Food
Level2 Concept: Nutrition
Output:
<output>Health</output>

Example 7:
Captions:
An adult standing behind a little girl while holding an umbrella.
The two people are standing under an umbrella in the rain.
Two girls under a large umbrella in the rain
A couple of people standing with a umbrella.
A child and person stand under an umbrella.
Level1 Concept: Parenting
Level2 Concept: Family
Output:
<output>Relationship</output>

Example 8:
Captions:
Woman snow boarding off of a cliff in the air.
A person doing a trick on a snowboard over a hill.
A snowboarder hitting a trick on a mountain.
a snowboarder in a black jacket is doing a trick
a person on a snow board flying in the air down the snow
Level1 Concept: Sport
Level2 Concept: Adventure
Output:
<output>Recreation</output>

Example 9:
Captions:
a person putting some pastries into a bag
A baker is placing their goods inside a bag.
A person working at a store selling pasties.
A person putting doughnuts into a bag in a shop.
Pans of potatoes and a person wearing a red apron in a commercial kitchen.
Level1 Concept: Commerce
Level2 Concept: Employment
Output:
<output>Service Industry</output>

Example 10:
Captions:
A laptop with a stationary mouse attached to it.
A laptop computer with an attached trackball device, showing the Windows logo on the screen.
The new Sony Y series laptop is connected to a special mouse.
A Windows laptop computer with a large mouse.
A Windows laptop with a corded mouse on top of a counter.
Level1 Concept: Technology
Level2 Concept: Innovation
Output:
<output>Productivity</output>

Example 11:
Captions:
A giraffe is peeking around the side of a wall at the camera.
A camel or a giraffe is playing with the camera-man.
A giraffe has its head pressed against the wall.
A half of face of a giraffe and a tree.
A giraffe standing next to a building near a tree.
Level1 Concept: Wildlife
Level2 Concept: Nature
Output:
<output>Animal Behavior</output>

Example 12:
Captions:
A walk in shower next to a toilet with a wooden seat.
The neat bathroom has green trim on the tile.
A toilet with a wood seat and a tiled floor.
This bathroom has a pattern of blue tiles on the floor.
A bathroom with blue tile outline around the floor.
Level1 Concept: Home Design
Level2 Concept: Aesthetics
Output:
<output>Personal Space</output>

Example 13:
Captions:
A pair of scissors sitting on top of a table.
A pair of scissors, tape, and wrapping paper lie on a wooden surface.
Some scrap book scissors are on a brown table.
a pair of scissor that is laying on a wooden table.
A pair of scissors and fabric on a wood table.
Level1 Concept: Crafting
Level2 Concept: Creativity
Output:
<output>Hobby</output>

Example 14:
Captions:
A fighter jet plane ascending into the sky.
A fighter jet gaining altitude in a cloudy sky.
An aircraft is flying in a cloudy sky.
A military jet taking off into the sky.
A nice jet plane flying into some grey skies.
Level1 Concept: Defense
Level2 Concept: Protection
Output:
<output>Safety</output>

Example 15:
Captions:
A man is in a kitchen making pizzas.
Man in apron standing on front of oven with pans and bakeware.
A baker is working in the kitchen rolling dough.
A person standing by a stove in a kitchen.
A table with pies being made and a person standing near a wall with pots and pans hanging on the wall.
Level1 Concept: Food Preparation
Level2 Concept: Cooking
Output:
<output>Domestic Activity</output>

Example 16:
Captions:
A parking meter next to a handicap parking space.
A parking meter that is placed right next to a parking lot.
a parking meter on the side of a road with a red light showing.
The parking meter in the parking lot has seven fifteen on it.
A parking meter sitting next to a parking space.
Level1 Concept: Urban Regulation
Level2 Concept: Infrastructure
Output:
<output>Urban Planning</output>

Example 17:
Captions:
THERE IS A GROUP OF PEOPLE SITTING AT A TABLE.
a group of people sitting at a long dining table in a restaurant.
A bunch of people sitting at a table having a discussion.
A large group of people at a table.
A group of people sitting around a table having a meal.
Level1 Concept: Social Gathering
Level2 Concept: Community
Output:
<output>Human Interaction</output>

Example 18:
Captions:
A young boy holding Nintendo Wii game controllers.
A little boy is staring attentively while playing a video game.
A small child holds two video game controllers.
A boy in grey shirt holding a Nintendo Wii controller.
a close up of a young child playing Nintendo Wii.
Level1 Concept: Technology
Level2 Concept: Digital Interaction
Output:
<output>Connectivity</output>

Example 19:
Captions:
View of a city skyline and train yard at dusk.
A city with skyscrapers, other buildings, and trains.
In a city, the buildings are illuminated while trains sit below.
An urban skyline with a train-yard in the foreground.
A city scape scene with a train yard in the foreground.
Level1 Concept: Urban Nightlife
Level2 Concept: Infrastructure
Output:
<output>Urbanization</output>

Example 20:
Captions:
a group of young people playing baseball in a field.
Baseball players are on the field during a game.
A young baseball player tagging up at third base.
five baseball players with a runner on the base.
Baseball players in their uniforms on a baseball field.
Level1 Concept: Sport
Level2 Concept: Competition
Output:
<output>Teamwork</output>

Example 21:
Captions:
Someone is driving a cart near a single engine airplane.
a small air plane on a small run way.
A small airplane that is parked near a run way.
A small airplane sitting next to an airport runway.
A crop dusting plane next to grassy field and tree.
Level1 Concept: Farming
Level2 Concept: Agriculture
Output:
<output>Food Production</output>

Example 22:
Captions:
There is a map in the street of the city.
a bus stop map in a city near a water fountain.
The subway stop Square Victoria entrance and the map of the neighborhood.
A map and street sign with building in background.
A map of the town in the middle of the street with buildings in the background.
Level1 Concept: Public Transport
Level2 Concept: Navigation
Output:
<output>Urban Planning</output>

Example 23:
Captions:
a green vase with some flowers and greenery.
A vase sitting next to a mirror filled with flowers.
This is a still life, slightly blurry, with a tea kettle and a floral arrangement.
A vase with roses and carnations in it by a mirror.
Flowers sit in a vase beside a tea pot.
Level1 Concept: Domestic Scene
Level2 Concept: Aesthetics
Output:
<output>Home Life</output>

Example 24:
Captions:
A utility truck is parked in the street beside traffic cones.
A street with orange cones and a work truck on it.
Traffic cones at the entrance of a lot where there is construction.
A construction truck is in front of the huge building.
a bunch of orange cones sitting in the road.
Level1 Concept: Construction
Level2 Concept: Urban Development
Output:
<output>Infrastructure</output>
"""
PROMPT_4 = """
Example 1:
Captions:
The tray on the bed has a pastry and two mugs on it.
A tray with coffee and a pastry on it.
A tray is full of breakfast foods and drinks on a bed.
coffee cream and a croissant on a tray
A tray on a bed with food and drink.
Level1 Concept: Breakfast
Level2 Concept: Meal
Level3 Concept: Food
Output:
<output>Nutrition</output>

Example 2:
Captions:
Man in dress shirt and orange tie standing inside a building.
a male with a beard and orange tie
A man wearing a neck tie and a white shirt.
A man posing for the picture in a building
A man dressed in a shirt and tie standing in a lobby.
Level1 Concept: Business Attire
Level2 Concept: Business
Level3 Concept: Career
Output:
<output>Personal Development</output>

Example 3:
Captions:
Many small children are posing together in the black and white photo.
A vintage school picture of grade school aged children.
A black and white photo of a group of kids.
A group of children standing next to each other.
A group of children standing and sitting beside each other.
Level1 Concept: Education
Level2 Concept: Learning
Level3 Concept: Personal Development
Output:
<output>Human Growth</output>

Example 4:
Captions:
there is a very tall giraffe standing in the wild
A giraffe is standing by some brush in a field.
A giraffe standing in a dirt field near a tree.
A single giraffe standing in a brushy area looking at the photographer
A giraffe standing in dry dead brush on the savannah.
Level1 Concept: Wildlife
Level2 Concept: Nature
Level3 Concept: Life
Output:
<output>Ecosystem</output>

Example 5:
Captions:
Ornate wedding cake ready at the hotel reception
A tall multi layer cake sitting on top of a blue table cloth.
A wedding cake with flowers in a banquet hall.
Nicely decorated three tier wedding cake with topper.
A very ornate, three layered wedding cake in a banquet room.
Level1 Concept: Wedding
Level2 Concept: Celebration
Level3 Concept: Tradition
Output:
<output>Cultural Heritage</output>

Example 6:
Captions:
A group of red, yellow and orange fruit mixed together.
a bunch of yellow and orange fruits in a pile
A bunch of yellow and orange fruit in varied sizes.
Pears and oranges have dirt and brown spots.
A picture of some fruit on a table.
Level1 Concept: Food
Level2 Concept: Nutrition
Level3 Concept: Health
Output:
<output>Well-being</output>

Example 7:
Captions:
An adult standing behind a little girl while holding an umbrella.
The two people are standing under an umbrella in the rain.
Two girls under a large umbrella in the rain
A couple of people standing with a umbrella.
A child and person stand under an umbrella.
Level1 Concept: Parenting
Level2 Concept: Family
Level3 Concept: Relationship
Output:
<output>Human Interaction</output>

Example 8:
Captions:
Woman snow boarding off of a cliff in the air.
A person doing a trick on a snowboard over a hill.
A snowboarder hitting a trick on a mountain.
a snowboarder in a black jacket is doing a trick
a person on a snow board flying in the air down the snow
Level1 Concept: Sport
Level2 Concept: Adventure
Level3 Concept: Recreation
Output:
<output>Human Activity</output>

Example 9:
Captions:
a person putting some pastries into a bag
A baker is placing their goods inside a bag.
A person working at a store selling pasties.
A person putting doughnuts into a bag in a shop.
Pans of potatoes and a person wearing a red apron in a commercial kitchen.
Level1 Concept: Commerce
Level2 Concept: Employment
Level3 Concept: Service Industry
Output:
<output>Business</output>

Example 10:
Captions:
A laptop with a stationary mouse attached to it.
A laptop computer with an attached trackball device, showing the Windows logo on the screen.
The new Sony Y series laptop is connected to a special mouse.
A Windows laptop computer with a large mouse.
A Windows laptop with a corded mouse on top of a counter.
Level1 Concept: Technology
Level2 Concept: Innovation
Level3 Concept: Productivity
Output:
<output>Development</output>

Example 11:
Captions:
A giraffe is peeking around the side of a wall at the camera.
A camel or a giraffe is playing with the camera-man.
A giraffe has its head pressed against the wall.
A half of face of a giraffe and a tree.
A giraffe standing next to a building near a tree.
Level1 Concept: Wildlife
Level2 Concept: Nature
Level3 Concept: Animal Behavior
Output:
<output>Environment</output>

Example 12:
Captions:
A walk in shower next to a toilet with a wooden seat.
The neat bathroom has green trim on the tile.
A toilet with a wood seat and a tiled floor.
This bathroom has a pattern of blue tiles on the floor.
A bathroom with blue tile outline around the floor.
Level1 Concept: Home Design
Level2 Concept: Aesthetics
Level3 Concept: Personal Space
Output:
<output>Living Space</output>

Example 13:
Captions:
A pair of scissors sitting on top of a table.
A pair of scissors, tape, and wrapping paper lie on a wooden surface.
Some scrap book scissors are on a brown table.
a pair of scissor that is laying on a wooden table.
A pair of scissors and fabric on a wood table.
Level1 Concept: Crafting
Level2 Concept: Creativity
Level3 Concept: Hobby
Output:
<output>Leisure</output>

Example 14:
Captions:
A fighter jet plane ascending into the sky.
A fighter jet gaining altitude in a cloudy sky.
An aircraft is flying in a cloudy sky.
A military jet taking off into the sky.
A nice jet plane flying into some grey skies.
Level1 Concept: Defense
Level2 Concept: Protection
Level3 Concept: Safety
Output:
<output>National Security</output>

Example 15:
Captions:
A man is in a kitchen making pizzas.
Man in apron standing on front of oven with pans and bakeware.
A baker is working in the kitchen rolling dough.
A person standing by a stove in a kitchen.
A table with pies being made and a person standing near a wall with pots and pans hanging on the wall.
Level1 Concept: Food Preparation
Level2 Concept: Cooking
Level3 Concept: Domestic Activity
Output:
<output>Human Behavior</output>

Example 16:
Captions:
A parking meter next to a handicap parking space.
A parking meter that is placed right next to a parking lot.
a parking meter on the side of a road with a red light showing.
The parking meter in the parking lot has seven fifteen on it.
A parking meter sitting next to a parking space.
Level1 Concept: Urban Regulation
Level2 Concept: Infrastructure
Level3 Concept: Urban Planning
Output:
<output>City Development</output>

Example 17:
Captions:
THERE IS A GROUP OF PEOPLE SITTING AT A TABLE.
a group of people sitting at a long dining table in a restaurant.
A bunch of people sitting at a table having a discussion.
A large group of people at a table.
A group of people sitting around a table having a meal.
Level1 Concept: Social Gathering
Level2 Concept: Community
Level3 Concept: Human Interaction
Output:
<output>Society</output>

Example 18:
Captions:
A young boy holding Nintendo Wii game controllers.
A little boy is staring attentively while playing a video game.
A small child holds two video game controllers.
A boy in grey shirt holding a Nintendo Wii controller.
a close up of a young child playing Nintendo Wii.
Level1 Concept: Technology
Level2 Concept: Digital Interaction
Level3 Concept: Connectivity
Output:
<output>Communication</output>

Example 19:
Captions:
View of a city skyline and train yard at dusk.
A city with skyscrapers, other buildings, and trains.
In a city, the buildings are illuminated while trains sit below.
An urban skyline with a train-yard in the foreground.
A city scape scene with a train yard in the foreground.
Level1 Concept: Urban Nightlife
Level2 Concept: Infrastructure
Level3 Concept: Urbanization
Output:
<output>Civilization</output>

Example 20:
Captions:
a group of young people playing baseball in a field.
Baseball players are on the field during a game.
A young baseball player tagging up at third base.
five baseball players with a runner on the base.
Baseball players in their uniforms on a baseball field.
Level1 Concept: Sport
Level2 Concept: Competition
Level3 Concept: Teamwork
Output:
<output>Human Interaction</output>

Example 21:
Captions:
Someone is driving a cart near a single engine airplane.
a small air plane on a small run way.
A small airplane that is parked near a run way.
A small airplane sitting next to an airport runway.
A crop dusting plane next to grassy field and tree.
Level1 Concept: Farming
Level2 Concept: Agriculture
Level3 Concept: Food Production
Output:
<output>Economy</output>

Example 22:
Captions:
There is a map in the street of the city.
a bus stop map in a city near a water fountain.
The subway stop Square Victoria entrance and the map of the neighborhood.
A map and street sign with building in background.
A map of the town in the middle of the street with buildings in the background.
Level1 Concept: Public Transport
Level2 Concept: Navigation
Level3 Concept: Urban Planning
Output:
<output>Mobility</output>

Example 23:
Captions:
a green vase with some flowers and greenery.
A vase sitting next to a mirror filled with flowers.
This is a still life, slightly blurry, with a tea kettle and a floral arrangement.
A vase with roses and carnations in it by a mirror.
Flowers sit in a vase beside a tea pot.
Level1 Concept: Domestic Scene
Level2 Concept: Aesthetics
Level3 Concept: Home Life
Output:
<output>Human Environment</output>

Example 24:
Captions:
A utility truck is parked in the street beside traffic cones.
A street with orange cones and a work truck on it.
Traffic cones at the entrance of a lot where there is construction.
A construction truck is in front of the huge building.
a bunch of orange cones sitting in the road.
Level1 Concept: Construction
Level2 Concept: Urban Development
Level3 Concept: Infrastructure
Output:
<output>Development</output>
"""