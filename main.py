import optparse

def getQValues():
    return


def parseOptions():
    optParser = optparse.OptionParser()
    optParser.add_option('-d', '--discount',action='store',
                         type='float',dest='discount',default=0.9,
                         help='Discount on future (default %default)')
    optParser.add_option('-r', '--livingReward',action='store',
                         type='float',dest='livingReward',default=0.0,
                         metavar="R", help='Reward for living for a time step (default %default)')
    optParser.add_option('-n', '--noise',action='store',
                         type='float',dest='noise',default=0.2,
                         metavar="P", help='How often action results in ' +
                         'unintended direction (default %default)' )
    optParser.add_option('-e', '--epsilon',action='store',
                         type='float',dest='epsilon',default=0.3,
                         metavar="E", help='Chance of taking a random action in q-learning (default %default)')
    optParser.add_option('-l', '--learningRate',action='store',
                         type='float',dest='learningRate',default=0.5,
                         metavar="P", help='TD learning rate (default %default)' )
    optParser.add_option('-i', '--iterations',action='store',
                         type='int',dest='iters',default=10,
                         metavar="K", help='Number of rounds of value iteration (default %default)')
    optParser.add_option('-k', '--episodes',action='store',
                         type='int',dest='episodes',default=1,
                         metavar="K", help='Number of epsiodes of the MDP to run (default %default)')
    optParser.add_option('-g', '--grid',action='store',
                         metavar="G", type='string',dest='grid',default="BookGrid",
                         help='Grid to use (case sensitive; options are BookGrid, BridgeGrid, CliffGrid, MazeGrid, default %default)' )
    optParser.add_option('-w', '--windowSize', metavar="X", type='int',dest='gridSize',default=150,
                         help='Request a window width of X pixels *per grid cell* (default %default)')
    optParser.add_option('-a', '--agent',action='store', metavar="A",
                         type='string',dest='agent',default="random",
                         help='Agent type (options are \'random\', \'value\' and \'q\', default %default)')
    optParser.add_option('-t', '--text',action='store_true',
                         dest='textDisplay',default=False,
                         help='Use text-only ASCII display')
    optParser.add_option('-p', '--pause',action='store_true',
                         dest='pause',default=False,
                         help='Pause GUI after each time step when running the MDP')
    optParser.add_option('-q', '--quiet',action='store_true',
                         dest='quiet',default=False,
                         help='Skip display of any learning episodes')
    optParser.add_option('-s', '--speed',action='store', metavar="S", type=float,
                         dest='speed',default=1.0,
                         help='Speed of animation, S > 1.0 is faster, 0.0 < S < 1.0 is slower (default %default)')
    optParser.add_option('-m', '--manual',action='store_true',
                         dest='manual',default=False,
                         help='Manually control agent')
    optParser.add_option('-v', '--valueSteps',action='store_true' ,default=False,
                         help='Display each step of value iteration')

    opts, args = optParser.parse_args()

    if opts.manual and opts.agent != 'q':
        print '## Disabling Agents in Manual Mode (-m) ##'
        opts.agent = None

    # MANAGE CONFLICTS
    if opts.textDisplay or opts.quiet:
    # if opts.quiet:
        opts.pause = False
        # opts.manual = False

    if opts.manual:
        opts.pause = True

    return opts

def runEpisode(agent, environment, discount):
    returns = 0
    totalDiscount = 1.0
    environment.reset()
    if 'startEpisode' in dir(agent):
        agent.startEpisode()
    print("beginning episode: {} \n".format(episode))
    while True:
        state = environment.getCurrentState()
        actions = environment.getPossibleActions(state)
        if not actions:
            print("Episode {} complete: return was {}\n".format(episode, returns))
            return returns

        # Get action
        action = agent.getAction(state)
        if action == None:
            raise 'Error: Agent returned None action'

        # Execute action
        nextState, reward = environment.doAction(action)
        print("Started in state: {}\n" 
              "Took action: {}\n"
              "Ended in state: {}\n"
              "Got reward: {}\n".format(state, action, nextState, reward))

        # Update learner
        if 'observeTransition' in dir(agent):
            agent.observeTransition(state, action, nextState, reward)

        returns += reward * totalDiscount
        totalDiscount *= discount

        if 'stopEpisode' in dir(agent):
            agent.stopEpisode()


if __name__ == "__main__":
    opts = parseOptions()
    ############################
    # Get the gridworld
    ############################
    import gridworld

    mdpFunction = getattr(gridworld, "get"+opts.grid)
    mdp = mdpFunction()
    mdp.setLivingReward(opts.livingReward)
    mdp.setNoise(opts.noise)
    env = gridworld.GridworldEnvironment(mdp)

    ############################
    # Get the Q-learning agent
    ############################
    import qlearningAgents

    actionFn = lambda state: mdp.getPossibleActions(state)
    qLearnOpts = {'gamma': opts.discount,
                  'alpha': opts.learningRate,
                  'epsilon': opts.epsilon,
                  'actionFn': actionFn}
    qAgent = qlearningAgents.QLearningAgent(**qLearnOpts)

    returns = 0
    for episode in range(1, opts.episodes+1):
        returns += runEpisode(qAgent, env, opts.discount)

    ############################
    # Get the Sarsa(0) agent
    ############################
    import sarsaAgents
    sarsaAgent = sarsaAgents.SarsaAgent(**qLearnOpts)

    returns = 0
    for episode in range(1, opts.episodes+1):
        returns += runEpisode(sarsaAgent, env, opts.discount)

    ############################
    # Get the Sarsa(lambda) agent
    ############################
    import sarsaLambdaAgents
    sarsaAgent = sarsaLambdaAgents.SarsaLambdaAgent(lamb=0.9, **qLearnOpts)

    returns = 0
    for episode in range(1, opts.episodes+1):
        returns += runEpisode(sarsaAgent, env, opts.discount)

    ############################
    # Get the approximate Sarsa(lambda) agent
    ############################
    import sarsaLambdaAgents
    sarsaAgent = sarsaLambdaAgents.ApproximateSarsaAgent(lamb=0.9, **qLearnOpts)

    returns = 0
    for episode in range(1, opts.episodes+1):
        returns += runEpisode(sarsaAgent, env, opts.discount)

