using Connectors.AI.LLamaSharp;
using Connectors.AI.LLamaSharp.TextCompletion;
using LLama;
using LLama.Common;
using Microsoft.SemanticKernel.AI.TextCompletion;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Microsoft.SemanticKernel.Connectors.AI.LLamaSharp.TextCompletion;

/// <summary>
/// Text Completion use LLamaSharp
/// </summary>
public sealed class LLamaSharpTextCompletion : ITextCompletion, IDisposable
{
    private readonly Func<LLamaWeights> _modelFunc;
    private readonly string _modelPath;
    private readonly ModelParams _params;
    private LLamaWeights? _model;


    /// <summary>
    /// Create LLamaSharpTextCompletion Instance
    /// </summary>
    /// <param name="modelPath"></param>
    public LLamaSharpTextCompletion(ModelParams @params)
    {
        this._params = @params;
        this._modelFunc = new Func<LLamaWeights>(() =>
        {
            return _model ??= LLamaWeights.LoadFromFile(_params); 
        });
    }

    public LLamaSharpTextCompletion(Func<LLamaWeights> modelFunc, ModelParams @params)
    {
        this._params = @params;
        this._modelFunc = modelFunc;
    }

    /// <summary>
    /// Dispose
    /// </summary>
    public void Dispose()
    {
        _model?.Dispose();
    }

    /// <inheritdoc/>
    public async Task<IReadOnlyList<ITextResult>> GetCompletionsAsync(string text, CompleteRequestSettings requestSettings, CancellationToken cancellationToken = default)
    {
        var executor = CreateExecutor();
        // TODO: InferAsync is not implemented in LLamaSharp, use Infer instead 
        var result = executor.Infer(text, requestSettings.ToLLamaSharpInferenceParams(), cancellationToken).ToAsyncEnumerable();
        return await Task.FromResult(new List<ITextResult> { new LLamaTextResult(result) }.AsReadOnly()).ConfigureAwait(false);
    }


    /// <inheritdoc/>
    public async IAsyncEnumerable<ITextStreamingResult> GetStreamingCompletionsAsync(string text, CompleteRequestSettings requestSettings, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var executor = CreateExecutor();
        // TODO: InferAsync is not implemented in LLamaSharp, use Infer instead 
        var result = executor.Infer(text, requestSettings.ToLLamaSharpInferenceParams(), cancellationToken).ToAsyncEnumerable();
        yield return new LLamaTextResult(result);
    }

    private StatelessExecutor CreateExecutor()
    {
        _model ??= this._modelFunc.Invoke();
        var context = _model.CreateContext(_params);
        return new(context);
    }
}
