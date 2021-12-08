rules_dict = \
    {'all' : ['fdgo', 'fdanti', 'delaygo', 'delayanti', 'dm1', 'dm2'] }

# Store indices of rules
rule_index_map = dict()
for ruleset, rules in rules_dict.items():
    rule_index_map[ruleset] = dict()
    for ind, rule in enumerate(rules):
        rule_index_map[ruleset][rule] = ind


def get_rule_index(rule, config):
    '''get the input index for the given rule'''
    return rule_index_map[config['ruleset']][rule] + config['rule_start']


def get_num_ring(ruleset):
    '''get number of stimulus rings'''
    return 3 if ruleset == 'oicdmc' else 2


def get_num_rule(ruleset):
    '''get number of rules'''
    return len(rules_dict[ruleset])
