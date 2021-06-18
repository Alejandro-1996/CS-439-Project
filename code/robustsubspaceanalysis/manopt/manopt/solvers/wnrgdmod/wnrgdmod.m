function [x, info, options] = wnrgdmod(problem, stepsize, x, options)
% based on manopt's steepest descent

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
    localdefaults.maxtime = 2;
    
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
    grad = getGradient(problem, x);
    gradnorm = problem.M.norm(x, grad);
    
    % Iteration counter.
    % At any point, iter is the number of fully executed iterations so far.
    iter = 0;
    
    % Save stats in a struct array info, and preallocate.
    stats = savestats();
    info(1) = stats;
    info(min(10000, options.maxiter+1)).iter = [];
    
    b = 1/stepsize;
    
    % Start iterating until stopping criterion triggers.
    while true
        
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
        newx = problem.M.retr(x, grad, -1/b);
        
        % Compute the new cost-related quantities for x
        newgrad = getGradient(problem, newx);%, storedb, newkey);
        newgradnorm = problem.M.norm(newx, newgrad);
        
        b = b + 1/b * newgradnorm;
        
        % Transfer iterate info, remove cache from previous x.
        %storedb.removefirstifdifferent(key, newkey);
        x = newx;
        %key = newkey;
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
        stats.gradnorm = gradnorm;
        if iter == 0
            stats.stepsize = NaN;
            stats.time = toc(timetic);
            stats.linesearch = [];
        else
            stats.stepsize = stepsize;
            stats.time = info(iter).time + toc(timetic);
            stats.linesearch = NaN;
        end
        stats.cost = getCost(problem, x);
        stats = applyStatsfun(problem, x, storedb, key, options, stats);
    end
    
end
