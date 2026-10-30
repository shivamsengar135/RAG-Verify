# 50 sampled RAG-Verify failures (Qwen2.5-7B, seed0-2)

## hotpotqa seed2 5abba4b4554299642a094af3 -- lost answer / answer-claim flag
- Q: What is the middle name of the Silicon Valley entrepreneur disussed in the non-fiction book The Unwinding?
- Gold: ['Andreas'] (in passages: True)
- First draft: 'Answer: Andreas\nExplanation: The passage states that Peter Thiel, whose full name is Peter Andreas Thiel, is one of the subjects discussed in "The Unwinding."'
- Final: "Answer: I don't know\nExplanation: The passages do not provide Peter Thiel's middle name."
- Answer-claim verdict: unsupported ('NOT_ENOUGH_INFO')

## hotpotqa seed2 5ae74fc45542997b22f6a6b4 -- wrong answer / retrieval miss
- Q: What is the full name of the viral respiratory disease in which Yi Guan's research helped to prevent the outbreak of?
- Gold: ['severe acute respiratory syndrome'] (in passages: False)
- First draft: "Answer: SARS\nExplanation: Yi Guan's research on the viral respiratory disease SARS allowed the Chinese government to successfully avert the 2004 outbreak of this disease."
- Final: "Answer: SARS\nExplanation: Yi Guan's research on the viral respiratory disease SARS allowed the Chinese government to successfully avert the 2004 outbreak of this disease."
- Answer-claim verdict: supported ('SUPPORTED')

## hotpotqa seed0 5a74bc1555429929fddd84dd -- lost answer / answer-claim flag
- Q: Which movie was released first, Fantastic Mr. Fox or Khan Kluay?
- Gold: ['Khan Kluay'] (in passages: True)
- First draft: 'Answer: Khan Kluay\nExplanation: Khan Kluay was released in 2006, while Fantastic Mr. Fox was released in 2009.'
- Final: 'Answer: Khan Kluay\nExplanation: Khan Kluay was released in 2006, while Fantastic Mr. Fox was released in 2009.'
- Answer-claim verdict: contradicted ('CONTRADICTED')

## hotpotqa seed1 5ae603a65542996de7b71af3 -- wrong answer / verifier accept
- Q: When was the Indian film actor, producer and playback singer born who debut with the 2000 War film, "Refugee" and had lead role in Main Prem Ki Diwani Hoon?
- Gold: ['5 February 1976'] (in passages: True)
- First draft: 'Answer: Abhishek Bachchan\nExplanation: Abhishek Bachchan made his screen debut with the 2000 War film "Refugee" and had a lead role in "Main Prem Ki Diwani Hoon".'
- Final: 'Answer: Abhishek Bachchan\nExplanation: Abhishek Bachchan made his screen debut with the 2000 War film "Refugee" and had a lead role in "Main Prem Ki Diwani Hoon".'
- Answer-claim verdict: supported ('SUPPORTED')

## hotpotqa seed1 5ae4903855429913cc2044a0 -- wrong answer / verifier accept
- Q: What do Woodstock and American Dream have in common?
- Gold: ['documentary film'] (in passages: True)
- First draft: 'Answer: Both are documentary films.\nExplanation: Woodstock is a documentary film about the 1969 music festival, and American Dream (1990 film) is described as a "cinéma vérité" documentary.'
- Final: 'Answer: Both are documentary films.\nExplanation: Woodstock is a documentary film about the 1969 music festival, and American Dream (1990 film) is described as a "cinéma vérité" documentary.'
- Answer-claim verdict: supported ('SUPPORTED')

## hotpotqa seed0 5ae72f75554299572ea5476c -- wrong answer / retrieval miss
- Q: The Black Star Line was a part of the Marcus Garvey movement that was also known by what names?
- Gold: ['Colonization movement or Black Zionism'] (in passages: False)
- First draft: 'Answer: Universal Negro Improvement Association and African Communities League (UNIA-ACL)\nExplanation: The passage states that Marcus Garvey founded the Universal Negro Improvement Association and African Communities League (UNIA-ACL), and the Black Star Line was a part of his movement.'
- Final: 'Answer: Universal Negro Improvement Association and African Communities League (UNIA-ACL)\nExplanation: The passage states that Marcus Garvey founded the Universal Negro Improvement Association and African Communities League (UNIA-ACL), and the Black Star Line was a part of his movement.'
- Answer-claim verdict: supported ('SUPPORTED')

## hotpotqa seed0 5a8ea03e5542990e94052b81 -- lost answer / answer-claim flag
- Q: Atlanta Zero Mile Post is located within a historic area opened in what year ?
- Gold: ['1969'] (in passages: True)
- First draft: 'Answer: 1969\nExplanation: The Atlanta Zero Mile Post is located within the Underground Atlanta Historic District, which was opened in 1969.'
- Final: "Answer: I don't know\nExplanation: The passages do not provide the year when the Underground Atlanta Historic District was opened."
- Answer-claim verdict: contradicted ('CONTRADICTED')

## hotpotqa seed2 5ade4fcc5542997c77adeded -- wrong answer / verifier accept
- Q: Multiculturalism in the Netherlands reached new heights following the murder in 2002, in part, with what Dutch politician, civil servant, socialogist and professor who formed his own party?
- Gold: ['Wilhelmus Simon Petrus Fortuijn, known as Pim Fortuyn'] (in passages: True)
- First draft: 'Answer: Pim Fortuyn\nExplanation: Pim Fortuyn, a Dutch politician who formed his own party in 2002, was murdered in 2002, which led to new heights in the political debate on multiculturalism in the Netherlands.'
- Final: 'Answer: Pim Fortuyn\nExplanation: Pim Fortuyn, a Dutch politician who formed his own party in 2002, was murdered in 2002, which led to new heights in the political debate on multiculturalism in the Netherlands.'
- Answer-claim verdict: supported ('SUPPORTED')

## hotpotqa seed0 5ab7e3c45542995dae37e9db -- wrong answer / verifier accept
- Q: Marine Air Control Group 48 is stationed at Naval Station Great Lakes which located where?
- Gold: ['Lake County, Illinois'] (in passages: True)
- First draft: 'Answer: Near North Chicago, Illinois\nExplanation: Naval Station Great Lakes, the location of Marine Air Control Group 48, is described in passage [3] as being near North Chicago, in Lake County, Illinois.'
- Final: 'Answer: Near North Chicago, Illinois\nExplanation: Naval Station Great Lakes, the location of Marine Air Control Group 48, is described in passage [3] as being near North Chicago, in Lake County, Illinois.'
- Answer-claim verdict: supported ('SUPPORTED')

## hotpotqa seed1 5a7ad9865542995eb53be8aa -- lost answer / answer-claim flag
- Q: The "civic rally cross" was sold alongside what other model sold exclusively in Japan at Honda Verno dealerships? 
- Gold: ['Honda Ballade'] (in passages: True)
- First draft: 'Answer: Honda Ballade\nExplanation: Both the Honda Ballade and the Civic Rally Cross were sold exclusively in Japan at Honda Verno dealerships.'
- Final: 'Answer: I don\'t know\nExplanation: The passages do not provide information about a "Civic Rally Cross" or its sales alongside any other model at Honda Verno dealerships.'
- Answer-claim verdict: unsupported ('NOT_ENOUGH_INFO')

## hotpotqa seed1 5ae3fcfc5542995dadf2428c -- wrong answer / verifier accept
- Q: When was the  American singer-songwriter born who recorded “Spirit In The Sky”?
- Gold: ['November 20, 1942'] (in passages: True)
- First draft: 'Answer: Norman Joel Greenbaum\nExplanation: According to passage [1], Norman Greenbaum was born on November 20, 1942, and is known for recording "Spirit in the Sky."'
- Final: 'Answer: Norman Joel Greenbaum\nExplanation: According to passage [1], Norman Greenbaum was born on November 20, 1942, and is known for recording "Spirit in the Sky."'
- Answer-claim verdict: supported ('SUPPORTED')

## hotpotqa seed0 5ae0198455429942ec259c1e -- wrong answer / verifier accept
- Q: How many times did the man who coached the 1986-87 UNLV Runnin' Rebels fail to win 20 games in a season?
- Gold: ['twice'] (in passages: True)
- First draft: "Answer: 0\nExplanation: Jerry Tarkanian, who coached the 1986–87 UNLV Runnin' Rebels, only failed to win 20 games in a season twice in his entire career, according to the passage."
- Final: "Answer: 0\nExplanation: Jerry Tarkanian, who coached the 1986–87 UNLV Runnin' Rebels, only failed to win 20 games in a season twice in his entire career, according to the passage."
- Answer-claim verdict: supported ('SUPPORTED')

## hotpotqa seed1 5a8a484f5542996c9b8d5e94 -- lost answer / answer-claim flag
- Q: What city did the author of "Portraits and Prayers" choose to live in after moving to France?
- Gold: ['Paris'] (in passages: True)
- First draft: 'Answer: Paris\nExplanation: While "Portraits and Prayers" is not directly mentioned in the passages, Gertrude Stein, who is associated with Paris in the provided text, is known for living in Paris after moving to France.'
- Final: 'Answer: I don\'t know\nExplanation: The passages do not mention the author of "Portraits and Prayers" or provide information about where they chose to live after moving to France.'
- Answer-claim verdict: unsupported ('NOT_ENOUGH_INFO')

## hotpotqa seed1 5a8a1b7a5542992d82986ec0 -- lost answer / answer-claim flag
- Q: Where did the father of Edward Garrison Walker live when he wrote "An Appeal to the Coloured Citizens of the World"?
- Gold: ['Boston'] (in passages: True)
- First draft: 'Answer: Boston\nExplanation: David Walker, the father of Edward Garrison Walker, published "An Appeal to the Coloured Citizens of the World" while living in Boston, Massachusetts.'
- Final: 'Answer: I don\'t know\nExplanation: The passages do not provide information about where David Walker, the father of Edward Garrison Walker, lived when he wrote "An Appeal to the Coloured Citizens of the World".'
- Answer-claim verdict: unsupported ('NOT_ENOUGH_INFO')

## hotpotqa seed0 5a8460d5554299123d8c2229 -- lost answer / answer-claim flag
- Q: Have singers Yang Yo-seob and Kim Tae-yeon both appeared in group bands?
- Gold: ['yes'] (in passages: False)
- First draft: "Answer: Yes\nExplanation: Yang Yo-seob is the main vocalist of the boy group Highlight, and Kim Tae-yeon is a member of the girl group Girls' Generation."
- Final: "Answer: Yes\nExplanation: Yang Yo-seob is the main vocalist of the boy group Highlight, and Kim Tae-yeon is a member of the girl group Girls' Generation."
- Answer-claim verdict: unsupported ('NOT_ENOUGH_INFO')

## hotpotqa seed2 5ac41379554299204fd21ef1 -- lost answer / answer-claim flag
- Q: Do Czesław Miłosz and Nathalie Sarraute have the same nationality?
- Gold: ['no'] (in passages: False)
- First draft: 'Answer: No\nExplanation: Czesław Miłosz was Polish, while Nathalie Sarraute was French.'
- Final: "Answer: I don't know\nExplanation: The passages provide information about Czesław Miłosz's Polish nationality and Nathalie Sarraute's French nationality, but they do not contain information about whether Czesław Miłosz and Nathalie Sarraute share the same nationality."
- Answer-claim verdict: contradicted ('CONTRADICTED')

## hotpotqa seed2 5ab85e195542992aa3b8c8c5 -- wrong answer / verifier accept
- Q: What profession does Alex Beard and Tony Hall, Baron Hall of Birkenhead have in common?
- Gold: ['director'] (in passages: True)
- First draft: 'Answer: Arts management\nExplanation: Both Alex Beard and Tony Hall, Baron Hall of Birkenhead, have held significant positions in the arts sector, with Beard as the chief executive of the Royal Opera House and Hall as the Director-General of the BBC, which oversees various arts and cultural programs.'
- Final: 'Answer: Arts management\nExplanation: Both Alex Beard and Tony Hall, Baron Hall of Birkenhead, have held significant positions in the arts sector, with Beard as the chief executive of the Royal Opera House and Hall as the Director-General of the BBC, which oversees various arts and cultural programs.'
- Answer-claim verdict: supported ('SUPPORTED')

## hotpotqa seed1 5a75d45f5542994ccc918630 -- lost answer / answer-claim flag
- Q: Which refers to more breeds, Stabyhoun or Griffon Bruxellois?
- Gold: ['Griffon Bruxellois'] (in passages: True)
- First draft: 'Answer: Griffon Bruxellois\nExplanation: The Griffon Bruxellois refers to three different breeds, while the Stabyhoun is a single breed.'
- Final: "Answer: I don't know\nExplanation: The passages do not provide information about the number of breeds that the Griffon Bruxellois refers to or the number of breeds that make up the Griffon type."
- Answer-claim verdict: unsupported ('NOT_ENOUGH_INFO')

## hotpotqa seed0 5ae28f43554299492dc91c2f -- wrong answer / verifier accept
- Q: Is Chicago Rockford International Airport or McClellan–Palomar Airport closer to their nearest cities?
- Gold: ['McClellan–Palomar Airport'] (in passages: True)
- First draft: 'Answer: Chicago Rockford International Airport\nExplanation: Chicago Rockford International Airport is located 4 miles south of Rockford, while McClellan–Palomar Airport is 3 miles southeast of Carlsbad, making it closer to its nearest city.'
- Final: 'Answer: Chicago Rockford International Airport\nExplanation: Chicago Rockford International Airport is located 4 miles south of Rockford, while McClellan–Palomar Airport is 3 miles southeast of Carlsbad, making it closer to its nearest city.'
- Answer-claim verdict: supported ('SUPPORTED')

## hotpotqa seed1 5adf41a45542993a75d26456 -- lost answer / answer-claim flag
- Q: In what year did Australian-born comedian Peter Helliar write I Love you Too?
- Gold: ['2010'] (in passages: True)
- First draft: 'Answer: 2010\nExplanation: According to passage [1], I Love You Too is described as a 2010 Australian romantic comedy film, and the screenplay was written by first-time writer Peter Helliar.'
- Final: "Answer: I don't know\nExplanation: The passages do not provide a specific year when Peter Helliar wrote the screenplay for I Love You Too."
- Answer-claim verdict: unsupported ('NOT_ENOUGH_INFO')

## hotpotqa seed2 5ac2d1d755429921a00ab053 -- lost answer / answer-claim flag
- Q: Who had the best singles ranking, Roberta Vinci or Jorge Lozano?
- Gold: ['Roberta Vinci'] (in passages: True)
- First draft: "Answer: Roberta Vinci\nExplanation: Roberta Vinci reached a career-high singles ranking of World No. 7, while Jorge Lozano's highest singles ranking was World No. 51."
- Final: "Answer: I don't know\nExplanation: The passages provide information about Roberta Vinci's and Jorge Lozano's highest singles rankings, but they do not contain enough information to determine who had the better singles ranking between the two."
- Answer-claim verdict: contradicted ('CONTRADICTED')

## hotpotqa seed1 5ae75a4c5542997b22f6a6f1 -- wrong answer / verifier accept
- Q: The Bart the Fink episode of The Simpsons aired in February of 1996, and involves the host of which kiddie show?
- Gold: ['The Itchy & Scratchy Show'] (in passages: True)
- First draft: 'Answer: Krusty the Clown\nExplanation: The passage states that in "Bart the Fink," Bart ruins Krusty the Clown\'s career after exposing him as a tax cheat, indicating Krusty is the host of a kiddie show.'
- Final: 'Answer: Krusty the Clown\nExplanation: The passage states that in "Bart the Fink," Bart ruins Krusty the Clown\'s career after exposing him as a tax cheat, indicating Krusty is the host of a kiddie show.'
- Answer-claim verdict: supported ('SUPPORTED')

## hotpotqa seed2 5ac1276c554299294b2190bc -- lost answer / answer-claim flag
- Q: Heinrich Gross participated in the Nazi regime's Aktion T4 more commonly referred to as what?
- Gold: ['Euthanasia Program'] (in passages: True)
- First draft: "Answer: Euthanasia Program\nExplanation: Heinrich Gross was involved in the killing of children under the Nazi regime's Euthanasia Program, which was also known as Aktion T4."
- Final: "Answer: Euthanasia Program\nExplanation: The passages refer to Aktion T4 as a part of the Nazi regime's mass murder through involuntary euthanasia, and it is also mentioned that Aktion T4 was a postwar name for the Euthanasia Program."
- Answer-claim verdict: unsupported ('NOT_ENOUGH_INFO')

## hotpotqa seed1 5a76a8bb5542993569682c76 -- lost answer / answer-claim flag
- Q: In what country did Gildo Siopaes win a bronze medal for bobsledding in the 1964 Winter Olympics?
- Gold: ['Austria'] (in passages: False)
- First draft: 'Answer: Austria\nExplanation: The 1964 Winter Olympics were held in Innsbruck, Austria.'
- Final: 'Answer: Austria\nExplanation: The 1964 Winter Olympics were held in Innsbruck, Austria, where Gildo Siorpaes won a bronze medal in the four-man bobsled event.'
- Answer-claim verdict: contradicted ('CONTRADICTED')

## hotpotqa seed2 5a75eda35542994ccc918661 -- wrong answer / verifier accept
- Q: How many consecutive years had the Serie A been comprised of 18 teams when Hernan Crespo got injured?
- Gold: ['15th consecutive'] (in passages: True)
- First draft: 'Answer: 15 years\nExplanation: The Serie A had been comprised of 18 teams for 15 consecutive years starting from the 1988-89 season, which continued through the 2002-03 season when Hernán Crespo got injured.'
- Final: 'Answer: 15 years\nExplanation: The Serie A had been comprised of 18 teams for 15 consecutive years starting from the 1988-89 season, which continued through the 2002-03 season.'
- Answer-claim verdict: supported ('SUPPORTED')

## triviaqa seed2 sfq_18832 -- lost answer / answer-claim flag
- Q: In which sport did Neil Adams win Olympic silver medals in 1980 and 1984?
- Gold: ['JUDO', 'Yuko (judo)', 'Zyuudou'] (in passages: True)
- First draft: 'Answer: Judo\nExplanation: Neil Adams won silver medals in the 1980 and 1984 Summer Olympic Games in judo, as stated in passage [1].'
- Final: "Answer: I don't know\nExplanation: The passages do not provide information about Neil Adams winning Olympic silver medals in 1980 and 1984 in any specific sport."
- Answer-claim verdict: unsupported ('NOT_ENOUGH_INFO')

## triviaqa seed1 tc_1623 -- wrong answer / retrieval miss
- Q: In Hanna and Barbera's TV cartoons base on The Addams Family who was the voice of Gomez?
- Gold: ['John Astin'] (in passages: False)
- First draft: 'Answer: Lennie Weinrib\nExplanation: According to passage [4], Lennie Weinrib voiced Gomez in the Hanna-Barbera animated series.'
- Final: 'Answer: Lennie Weinrib\nExplanation: According to passage [4], Lennie Weinrib voiced Gomez in the Hanna-Barbera animated series.'
- Answer-claim verdict: supported ('SUPPORTED')

## triviaqa seed1 qz_3903 -- lost answer / answer-claim flag
- Q: In which postcode area are the Wimbledon Championships held? SW15, SW17 or SW19?
- Gold: ['SW19', 'SW15', 'SW14'] (in passages: True)
- First draft: 'Answer: SW15\nExplanation: The passage mentions that SW15 corresponds to the Putney district, which includes the location of the Wimbledon Championships.'
- Final: "Answer: I don't know\nExplanation: The passages do not provide information about the location of the Wimbledon Championships or which postcode area they are held in."
- Answer-claim verdict: unsupported ('NOT_ENOUGH_INFO')

## triviaqa seed1 qw_2462 -- wrong answer / near miss
- Q: Which international treaties of 1899 and 1907 cover the use of weapons in war?
- Gold: ['Hague Conventions', 'Hague conventions', 'The Hague Convention'] (in passages: True)
- First draft: 'Answer: The Hague Declarations\nExplanation: The passages mention that the Hague Conventions of 1899 and 1907 outlawed the use of dangerous chemical agents.'
- Final: 'Answer: The Hague Declarations\nExplanation: The passages mention that the Hague Conventions of 1899 and 1907 outlawed the use of dangerous chemical agents.'
- Answer-claim verdict: supported ('SUPPORTED')

## triviaqa seed2 qb_7241 -- lost answer / answer-claim flag
- Q: The largest known volcano in our solar system is on which planet?
- Gold: ['Mars', 'Fourth planet', 'Sol 4'] (in passages: True)
- First draft: 'Answer: Mars\nExplanation: Mars hosts the largest known volcanoes in the solar system, including Olympus Mons.'
- Final: 'Answer: Mars\nExplanation: The passages state that Mars has four vast shield volcanoes, one of which, Olympus Mons, is the largest known volcano in the solar system.'
- Answer-claim verdict: contradicted ('CONTRADICTED')

## triviaqa seed1 qw_243 -- lost answer / answer-claim flag
- Q: The main publicity image for which Alfred Hitchcock film featured Cary Grant being chased by a bi-plane?
- Gold: ['North By Northwest', 'N by nw', "The Man in Lincoln's Nose"] (in passages: True)
- First draft: 'Answer: North by Northwest\nExplanation: The passage mentions that Cary Grant starred in North by Northwest, playing an advertising executive who becomes embroiled in a case of mistaken identity, which aligns with the description of being chased by a bi-plane.'
- Final: "Answer: I don't know\nExplanation: The passages do not provide information about the main publicity image for any Alfred Hitchcock film, let alone one featuring Cary Grant being chased by a bi-plane."
- Answer-claim verdict: unsupported ('NOT_ENOUGH_INFO')

## triviaqa seed0 bt_3739 -- wrong answer / verifier accept
- Q: What pattern would be on material described as Tattersall
- Gold: ['Checks', 'Check (disambiguation)', 'Checked'] (in passages: True)
- First draft: 'Answer: Tattersall pattern\nExplanation: The Tattersall pattern is described as a check or plaid pattern woven into cloth, characterized by regularly-spaced thin, even vertical warp stripes repeated horizontally in the weft, forming squares.'
- Final: 'Answer: Tattersall pattern\nExplanation: The Tattersall pattern is described as a check or plaid pattern woven into cloth, characterized by regularly-spaced thin, even vertical warp stripes repeated horizontally in the weft, forming squares.'
- Answer-claim verdict: supported ('SUPPORTED')

## triviaqa seed2 qz_4443 -- lost answer / answer-claim flag
- Q: How is Frances Gumm the actress better known?
- Gold: ['Judy Garland', 'Joey Luft', 'Virgina gumm'] (in passages: False)
- First draft: 'Answer: Judy Garland\nExplanation: Frances Gumm was the stage name of the famous American actress and singer Judy Garland, who is better known by her professional name.'
- Final: "Answer: I don't know\nExplanation: The passages do not contain any information about Frances Gumm or Judy Garland."
- Answer-claim verdict: unsupported ('NOT_ENOUGH_INFO')

## triviaqa seed0 bb_6033 -- lost answer / answer-claim flag
- Q: Film musical interpretations of Romeo and Juliet and the memoir of Baroness von Trapp shared what lead character name?
- Gold: ['Maria', 'María', 'Maria (singer)'] (in passages: True)
- First draft: 'Answer: Maria\nExplanation: In "The Sound of Music," the lead character is named Maria, while in "Romeo and Juliet," there is no lead character named Maria.'
- Final: 'Answer: I don\'t know\nExplanation: The passages do not provide information about a lead character named Maria in "Romeo and Juliet."'
- Answer-claim verdict: unsupported ('NOT_ENOUGH_INFO')

## triviaqa seed1 sfq_8219 -- wrong answer / retrieval miss
- Q: In which country was the 2010 football African Nations Cup held?
- Gold: ['ANGOLA', 'Angola', 'Angloa'] (in passages: False)
- First draft: 'Answer: South Africa\nExplanation: The 2010 Africa Cup of Nations was held in South Africa, as stated in passage [2].'
- Final: 'Answer: South Africa\nExplanation: The 2010 Africa Cup of Nations was held in South Africa, as stated in passage [2].'
- Answer-claim verdict: supported ('SUPPORTED')

## triviaqa seed1 qz_5409 -- wrong answer / retrieval miss
- Q: Rarely is the question asked: Is our children learning?
- Gold: ['George W Bush', 'George Walker Bush', 'Dubya'] (in passages: False)
- First draft: 'Answer: Bush mentioned education.\nExplanation: The quote directly relates to the topic of education, specifically questioning the focus on whether children are learning.'
- Final: 'Answer: Bush mentioned education.\nExplanation: The quote directly relates to the topic of education, specifically questioning the focus on whether children are learning.'
- Answer-claim verdict: supported ('SUPPORTED')

## triviaqa seed0 qg_1091 -- lost answer / answer-claim flag
- Q: Who's missing: Miss Scarlett, Mrs. White, Mr. Green, Mrs. Peacock, Professor Plum?
- Gold: ['Colonel Mustard', 'List of Cluedo characters', 'Miss Scarlett'] (in passages: True)
- First draft: 'Answer: Mrs. White\nExplanation: Mrs. White is not mentioned in any of the provided passages, while the other characters are discussed in detail.'
- Final: "Answer: I don't know\nExplanation: The passages do not provide information about Mrs. White, nor do they list all the characters in Cluedo, making it impossible to determine if Mrs. White is missing."
- Answer-claim verdict: unsupported ('NOT_ENOUGH_INFO')

## triviaqa seed2 tb_1490 -- lost answer / answer-claim flag
- Q: What is the name of the vessel in Paul Gallico's 1969 adventure novel that has been adapted as a film four times?
- Gold: ['Poseidon', 'Earth Shaker', 'God Of The Seas'] (in passages: True)
- First draft: 'Answer: The Poseidon Adventure\nExplanation: The passage mentions that Paul Gallico wrote "The Poseidon Adventure," which was adapted into a film in 1972 and has been adapted four times according to the information provided.'
- Final: "Answer: I don't know\nExplanation: The passages do not provide information about a 1969 adventure novel by Paul Gallico or the number of film adaptations of any of his works."
- Answer-claim verdict: contradicted ('CONTRADICTED')

## triviaqa seed1 wh_2109 -- wrong answer / verifier accept
- Q: The name of which chemical element of the Periodic Table ends with the letters KEL ?
- Gold: ['NICKEL', 'Nickel metallurgy', 'Nickel mining in BC'] (in passages: True)
- First draft: 'Answer: Potassium\nExplanation: The element name "potassium" ends with "KEL" when written in its chemical symbol form, "K".'
- Final: 'Answer: Potassium\nExplanation: The element name "potassium" ends with "KEL" when written in its chemical symbol form, "K".'
- Answer-claim verdict: supported ('SUPPORTED')

## triviaqa seed1 qz_4575 -- wrong answer / verifier accept
- Q: Which dancer has only one notable film cameo, appearing with Sean Connery in the 1979 movie The First Great Train Robbery, breaking into a guarded station office.
- Gold: ['Wayne Sleep'] (in passages: True)
- First draft: 'Answer: Clean Willy\nExplanation: Clean Willy, played by Wayne Sleep, is described as a cat burglar who climbs down from the ceiling and opens the office doors from within to break into the guarded station office.'
- Final: 'Answer: Clean Willy\nExplanation: Clean Willy, played by Wayne Sleep, is described as a cat burglar who climbs down from the ceiling and opens the office doors from within to break into the guarded station office.'
- Answer-claim verdict: supported ('SUPPORTED')

## triviaqa seed1 sfq_19992 -- lost answer / answer-claim flag
- Q: "A dish cooked ""a la Lyonnaise"" would be garnished or prepared with which ingredient?"
- Gold: ['Onions', 'Onion skins', 'Allium angolense'] (in passages: True)
- First draft: 'Answer: onions\nExplanation: While not explicitly stated in the passages, a "la Lyonnaise" preparation typically involves onions, which is a common garnish or ingredient in French cooking, especially in Lyon, France.'
- Final: 'Answer: I don\'t know\nExplanation: The passages do not contain information about "a la Lyonnaise" preparation or its associated ingredients.'
- Answer-claim verdict: unsupported ('NOT_ENOUGH_INFO')

## triviaqa seed2 tb_1129 -- lost answer / support-claim flag
- Q: The flag of which country prominently features 'The Eagle of Saladin' holding a scroll bearing the name of the country in Arabic?
- Gold: ['Egypt', 'EGY', 'Arab Republic of Egypt'] (in passages: True)
- First draft: "Answer: Egypt\nExplanation: The passage states that the Eagle of Saladin is used as part of the emblems of Egypt, and it is associated with the country's coat of arms and flags."
- Final: "Answer: I don't know\nExplanation: The passages do not provide information about a scroll bearing the name of the country in Arabic being held by the Eagle of Saladin."
- Answer-claim verdict: supported ('SUPPORTED')

## triviaqa seed1 qb_9527 -- wrong answer / verifier accept
- Q: Who was the first person to bear the title of Holy Roman Emperor?
- Gold: ['Charlemagne', 'Gisela, daughter of Charlemagne', 'Regina (concubine)'] (in passages: True)
- First draft: 'Answer: Henry VII of the House of Luxembourg\nExplanation: According to passage [1], Henry VII of the House of Luxembourg was crowned as the first Holy Roman Emperor since Frederick II in 1312.'
- Final: 'Answer: Henry VII of the House of Luxembourg\nExplanation: According to passage [1], Henry VII of the House of Luxembourg was crowned as the first Holy Roman Emperor since Frederick II in 1312.'
- Answer-claim verdict: supported ('SUPPORTED')

## triviaqa seed1 sfq_14137 -- wrong answer / verifier accept
- Q: If you were in Japan eating edamame, what would you be eating?
- Gold: ['(Soy) Beans', 'Soy Isoflavones', 'Soya Bean'] (in passages: True)
- First draft: 'Answer: Edamame\nExplanation: Edamame is a preparation of immature soybeans in the pod, commonly found in Japanese cuisine and often served as a side dish with salt.'
- Final: 'Answer: Edamame\nExplanation: Edamame is a preparation of immature soybeans in the pod, commonly found in Japanese cuisine and often served as a side dish with salt.'
- Answer-claim verdict: supported ('SUPPORTED')

## triviaqa seed0 sfq_22043 -- lost answer / answer-claim flag
- Q: The Dales Way is a path that runs from Bowness in Cumbria to which town in Yorkshire on the River Wharfe?
- Gold: ['ILKLEY', 'Bingley College', 'Ilkley College'] (in passages: True)
- First draft: 'Answer: Ilkley\nExplanation: The Dales Way starts in Ilkley, West Yorkshire, and ends in Bowness-on-Windermere, Cumbria.'
- Final: "Answer: I don't know\nExplanation: The passages do not provide enough information to determine the specific town on the River Wharfe that the Dales Way ends at."
- Answer-claim verdict: contradicted ('CONTRADICTED')

## triviaqa seed2 dpql_4988 -- wrong answer / verifier accept
- Q: What was Margot’s surname in “The Good Life” TV series?
- Gold: ['Leadbetter', 'Leadbetter (disambiguation)'] (in passages: True)
- First draft: 'Answer: Sturgess\nExplanation: In the passage, it states "Penelope Keith — Margo Leadbetter (née Sturgess)," indicating Margo\'s maiden name before marriage to Jeremy Leadbetter.'
- Final: 'Answer: Sturgess\nExplanation: In the passage, it states "Penelope Keith — Margo Leadbetter (née Sturgess)," indicating Margo\'s maiden name before marriage to Jeremy Leadbetter.'
- Answer-claim verdict: supported ('SUPPORTED')

## triviaqa seed1 sfq_13715 -- wrong answer / retrieval miss
- Q: In 2011, who became the first artist to win the Mercury Music Prize twice, first winning it in 2001?
- Gold: ['P J HARVEY', 'PJH', 'Polly Harvey'] (in passages: False)
- First draft: "Answer: Elbow\nExplanation: The passage states that Elbow won the Mercury Prize in 2008 and mentions that she (presumably referring to Elbow, a band, as artists are often referred to as 'she' in such contexts) was the first female solo artist to receive the award and tied with Radiohead as the most shortlisted artist, with Radiohead having never won, implying Elbow won in 2001 and again in 2008."
- Final: "Answer: Elbow\nExplanation: The passage states that Elbow won the Mercury Prize in 2008 and mentions that she (presumably referring to Elbow, a band, as artists are often referred to as 'she' in such contexts) was the first female solo artist to receive the award and tied with Radiohead as the most shortlisted artist, with Radiohead having never won, implying Elbow won in 2001 and again in 2008."
- Answer-claim verdict: supported ('SUPPORTED')

## triviaqa seed1 tc_1421 -- lost answer / answer-claim flag
- Q: What is Harrison Ford's real name?
- Gold: ['Harrison Ford', 'Harrison Ford (actor)', 'Harrison ford'] (in passages: True)
- First draft: 'Answer: Harrison Ford\nExplanation: The passages consistently refer to the individual as Harrison Ford without providing a different name.'
- Final: "Answer: I don't know\nExplanation: The passages do not provide Harrison Ford's real name or any other name he might have used."
- Answer-claim verdict: unsupported ('NOT_ENOUGH_INFO')

## triviaqa seed1 bb_2583 -- wrong answer / verifier accept
- Q: Near Field Communication technology (NFC) is a set of standards for?
- Gold: ['Smartphones', 'SmartPhone', 'Smart-phones'] (in passages: True)
- First draft: 'Answer: a set of communication protocols\nExplanation: NFC is defined as a set of communication protocols that enable electronic devices to establish communication by bringing them within close proximity of each other.'
- Final: 'Answer: a set of communication protocols\nExplanation: NFC is defined as a set of communication protocols that enable electronic devices to establish communication by bringing them within close proximity of each other.'
- Answer-claim verdict: supported ('SUPPORTED')

## triviaqa seed1 sfq_20557 -- wrong answer / retrieval miss
- Q: Who was the conservative candidate beaten by Ken Livingstone in the first two elections for Mayor of London?
- Gold: ['Steven Norris', 'Steve Norris', 'Steven John Norris'] (in passages: False)
- First draft: 'Answer: Boris Johnson\nExplanation: According to passage [3], Boris Johnson defeated Ken Livingstone in the 2008 London mayoral election. Passage [4] confirms that Livingstone was defeated by Johnson again in the 2012 Mayoral Elections.'
- Final: 'Answer: Boris Johnson\nExplanation: According to passage [3], Boris Johnson defeated Ken Livingstone in the 2008 London mayoral election. Passage [4] confirms that Livingstone was defeated by Johnson again in the 2012 Mayoral Elections.'
- Answer-claim verdict: supported ('SUPPORTED')
