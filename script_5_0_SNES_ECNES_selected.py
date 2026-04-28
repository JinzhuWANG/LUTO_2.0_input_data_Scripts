'''
Species and ecological community target lists for NECMA and GBCMA NRM regions.
Imported by 5_2_get_NVIS_SNES_ECNES_targets_by_regions.py.

All species/communities listed here are from the NECMA/Deakin contract.
Species absent from the DCCEEW SNES database are included but will silently
produce no matches when used as isin() filters.
'''

NECMA_SNES = [
    'Euphrasia eichleri', 'Grevillea burrowa', 'Glycine latrobeana',
    'Pomaderris subplicata', 'Caladenia concolor', 'Pterostylis X aenigma',
    'Sannantha crenulata', 'Grevillea jephcottii', 'Zieria citriodora',
    'Banksia canei', 'Acacia phasmoides', 'Argyrotegium nitidulum',
    'Kelleria bogongensis', 'Lobelia gelida', 'Euphrasia crassiuscula subsp. glandulifera',
    'Eucalyptus cadens', 'Rostratula australis', 'Ninox connivens',
    'Burhinus grallarius', 'Anthochaera phrygia', 'Lathamus discolor',
    'Mastacomys fuscus mordicus', 'Potorous longipes', 'Burramys parvus',
    'Ornithorhynchus anatinus', 'Pseudomys fumeus', 'Petauroides volans',
    'Dasyurus maculatus maculatus', 'Litoria verreauxii alpina',
    'Litoria booroolongensis', 'Crinia sloanei', 'Litoria spenceri',
    'Pseudemoia cryodroma', 'Cyclodomorphus praealtus', 'Vermicella annulata',
    'Liopholis guthega', 'Morelia spilota metcalfei', 'Thaumatoperla alpina',
    'Synemon plana', 'Keyacris scurra', 'Galaxias rostratus',
    'Macquaria australasica', 'Maccullochella peelii', 'Nannoperca australis',
    'Maccullochella macquariensis',
]

GBCMA_SNES = [
    'Galaxias rostratus', 'Bidyanus bidyanus', 'Galaxias fuscus',
    'Maccullochella macquariensis', 'Macquaria australasica', 'Maccullochella peelii',
    'Nannoperca australis Murray-Darling Basin lineage', 'Lathamus discolor', 'Anthochaera phrygia',
    'Gymnobelideus leadbeateri', 'Litoria spenceri', 'Pomaderris vacciniifolia',
    'Pimelea spinescens subsp. spinescens', 'Botaurus poiciloptilus', 'Burramys parvus',
    'Senecio behrianus', 'Eucalyptus alligatrix subsp. limaensis', 'Eucalyptus crenulata',
    'Polytelis swainsonii', 'Litoria raniformis', 'Pteropus poliocephalus',
    'Falco hypoleucos', 'Hirundapus caudacutus', 'Grantiella picta',
    'Delma impar', 'Synemon plana', 'Melanodryas cucullata',
    'Calochilus richiae', 'Swainsona recta', 'Sclerolaena napiformis',
    'Euphrasia collina subsp. muelleri', 'Dianella amoena', 'Glycine latrobeana',
    'Caladenia concolor', 'Hibbertia humifusa subsp. erigens', 'Rostratula australis',
    'Crinia sloanei', 'Calidris ferruginea', 'Brachyscome muelleroides',
    'Myriophyllum porcatum', 'Swainsona murrayana', 'Swainsona plagiotropis',
    'Amphibromus fluitans', 'Lepidium monoplocoides', 'Callocephalon fimbriatum',
    'Dasyurus maculatus maculatus', 'Petauroides volans', 'Pseudomys fumeus',
    'Liopholis montana', 'Pycnoptilus floccosus', 'Petaurus australis',
    'Mastacomys fuscus mordicus',
]

NECMA_ECNES = [
    'Alpine Sphagnum Bogs and Associated Fens',
    'Buloke Woodlands of the Riverina and Murray-Darling Depression Bioregions',
    'Grey Box (Eucalyptus microcarpa) Grassy Woodlands and Derived Native Grasslands of South-eastern Australia',
    "White Box-Yellow Box-Blakely's Red Gum Grassy Woodland and Derived Native Grassland",
]

GBCMA_ECNES = [
    'Seasonal Herbaceous Wetlands (Freshwater) of the Temperate Lowland Plains',
    "White Box-Yellow Box-Blakely's Red Gum Grassy Woodland and Derived Native Grassland",
    'Natural Grasslands of the Murray Valley Plains',
    'Alpine Sphagnum Bogs and Associated Fens',
    'Grey Box (Eucalyptus microcarpa) Grassy Woodlands and Derived Native Grasslands of South-eastern Australia',
    'Buloke Woodlands of the Riverina and Murray-Darling Depression Bioregions',
]

SNES_AUSTRALIA  = list(set(NECMA_SNES  + GBCMA_SNES))
ECNES_AUSTRALIA = list(set(NECMA_ECNES + GBCMA_ECNES))
