class CRPMixtureProblem(stochastic.StochasticInferenceProblem):
    def sample_latents(self, state, params):
        # SET SEED
        numpy.random.seed(params['latent_seed'])

        # check that we've supplied a dataset size
        if 'N' not in params:
            raise Exception("invalid CRP parameters: no N provided")

        # check that we've supplied a dimensionality
        if 'D' not in params:
            raise Exception("invalide CRP parameters: no D provided")

        # check that we've supplied types for each dimension
        if 'D_types' not in params:
            raise Exception("invalid CRP params: no D_types provided")

        # FIXME: generalize past beta-bernoulli throughout
        for d_type in params['D_types']:
            if d_type is not "binary":
                raise Exception("unknown CRP D_type: " + str(d_type))

        # sample concentration parameter or use fixed value
        if 'alpha' in params:
            state['alpha'] = params['alpha']
        else:
            state['alpha'] = scipy.stats.gamma(1, scale=1)

        state['tables'] = []

        # sample cluster assignments
        for i in range(params['N']):
            if i == 0:
                state['tables'].append([0])
            else:
                p = numpy.random.random()
                chosen_table = 0
                for table in state['tables']:
                    table_prob = (float(len(table)) / float(i + state['alpha']))
                    if p < table_prob
                        break
                    chosen_table += 1
                    p -= table_prob
                if p < 0:
                    state['tables'][chosen_table].append(i)
                else:
                    state['tables'].append([i])
        
        # sample cluster hyper-parameters
        state['component_hypers'] = [scipy.stats.gamma(1, scale=1) for d in range(params['D'])]

        # initialize empty sufficient statistics
        for d in range(params['D']):
            state['component_suff_stats'] = [[0, 0] for c in range(len(state['tables']))]

    def evaluate_log_joint_latents(self, state, params):
        out = 0.0

        #concentration
        std_gamma = scipy.stats.gamma(1, scale=1)
        out += std_gamma.logpdf(state['alpha'])

        #components
        for d in range(params['D']):
            out += std_gamma.logpdf(state['component_hypers'][d])
            
        #crp
        out += len(state['tables']) * numpy.log(state['alpha'])
        for table in state['tables']:
            out += scipy.special.gammaln(len(table))
        out += scipy.special.gammaln(state['alpha'])
        out -= scipy.special.gammaln(state['alpha'] + params['N'])

        return out

    def sample_observables(self, state, params):
        # FIXME: Generalize to other data types
        
        for state in 
