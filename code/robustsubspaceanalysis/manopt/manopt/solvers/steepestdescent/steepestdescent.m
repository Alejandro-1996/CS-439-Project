function [x, cost, info, options] = steepestdescent(problem, x, options)
    
    % Verify that the problem description is sufficient for the solver.
    if ~canGetCost(problem)
        warning('manopt:getCost', ...
                'No cost provided. The algorithm will likely abort.');
    end
    if ~canGetGradient(problem) && ~canGetApproxGradient(problem)
        % Note: we do not give a warning if an approximate gradient is
        % explicitly given in the problem description, as in that case the
        % user seems to be aware of the issue.
        warning('manopt:getGradient:approx', ...
               ['No gradient provided. Using an FD approximation instead (slow).\n' ...
                'It may be necessary to increase options.tolgradnorm.\n' ...
                'To disable this warning: warning(''off'', ''manopt:getGradient:approx'')']);
        problem.approxgrad = approxgradientFD(problem);
    end
    
    % Set local defaults here.
    localdefaults.minstepsize = 1e-10;
    localdefaults.maxiter = 1000;
    localdefaults.tolgradnorm = 1e-6;
    
    % Depending on whether the problem structure specifies a hint for
    % line-search algorithms, choose a default line-search that works on
    % its own (typical) or that uses the hint.
    if ~canGetLinesearch(problem)
        localdefaults.linesearch = @linesearch;
    else
        localdefaults.linesearch = @linesearch_hint;
    end
    
    % Merge global and local defaults, then merge w/ user options, if any.
    localdefaults = mergeOptions(getGlobalDefaults(), localdefaults);
    if ~exist('options', 'var') || isempty(options)
        options = struct();
    end
    options = mergeOptions(localdefaults, options);
    
    timetic = tic();
    
    % If no initial point x is given by the user, generate one at random.
    if ~exist('x', 'var') || isempty(x)
        x = problem.M.rand();
    end
    
    % Create a store database and get a key for the current x.
    storedb = StoreDB(options.storedepth);
    key = storedb.getNewKey();
    
    % Compute objective-related quantities for x.
    [cost, grad] = getCostGrad(problem, x);
    gradnorm = problem.M.norm(x, grad);
    
    % Iteration counter.
    % At any point, iter is the number of fully executed iterations so far.
    iter = 0;
    
    % Save stats in a struct array info, and preallocate.
    stats = savestats();
    info(1) = stats;
    info(min(10000, options.maxiter+1)).iter = [];
    
    if options.verbosity >= 2
        fprintf(' iter\t               cost val\t    grad. norm\n');
    end
    
    % Start iterating until stopping criterion triggers.
    while true

        % Display iteration information.
        if options.verbosity >= 2
            fprintf('%5d\t%+.16e\t%.8e\n', iter, cost, gradnorm);
        end
        
        % Start timing this iteration.
        timetic = tic();
        
        % Run standard stopping criterion checks.
        [stop, reason] = stoppingcriterion(problem, x, options, ...
                                                             info, iter+1);
        
        % If none triggered, run specific stopping criterion check.
        if ~stop && stats.stepsize < options.minstepsize
            stop = true;
            reason = sprintf(['Last stepsize smaller than minimum '  ...
                              'allowed; options.minstepsize = %g.'], ...
                              options.minstepsize);
        end
    
        if stop
            if options.verbosity >= 1
                fprintf([reason '\n']);
            end
            break;
        end

        % Pick the descent direction as minus the gradient.
        desc_dir = problem.M.lincomb(x, -1, grad);
        
        % Execute the line search.
        [stepsize, newx, newkey, lsstats] = options.linesearch( ...
                             problem, x, desc_dir, cost, -gradnorm^2, ...
                             options);
        
        % Compute the new cost-related quantities for x
        [newcost, newgrad] = getCostGrad(problem, newx);
        newgradnorm = problem.M.norm(newx, newgrad);
        
        % Transfer iterate info, remove cache from previous x.
        %storedb.removefirstifdifferent(key, newkey);
        x = newx;
        key = newkey;
        cost = newcost;
        grad = newgrad;
        gradnorm = newgradnorm;
        
        % Make sure we don't use too much memory for the store database.
        storedb.purge();
        
        % iter is the number of iterations we have accomplished.
        iter = iter + 1;
        
        % Log statistics for freshly executed iteration.
        stats = savestats();
        info(iter+1) = stats;
        
    end
    
    
    info = info(1:iter+1);

    if options.verbosity >= 1
        fprintf('Total time is %f [s] (excludes statsfun)\n', ...
                info(end).time);
    end
    
    
    
    % Routine in charge of collecting the current iteration stats
    function stats = savestats()
        stats.iter = iter;
        stats.cost = cost;
        stats.gradnorm = gradnorm;
        if iter == 0
            stats.stepsize = NaN;
            stats.time = toc(timetic);
            stats.linesearch = [];
        else
            stats.stepsize = stepsize;
            stats.time = info(iter).time + toc(timetic);
            stats.linesearch = lsstats;
        end
        stats = applyStatsfun(problem, x, storedb, key, options, stats);
    end
    
end
