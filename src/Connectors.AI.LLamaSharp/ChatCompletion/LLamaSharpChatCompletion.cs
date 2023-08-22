﻿using LLama;
using Microsoft.SemanticKernel.AI.ChatCompletion;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Connectors.AI.LLamaSharp.ChatCompletion
{
    /// <summary>
    /// LLamaSharp ChatCompletion
    /// </summary>
    public sealed class LLamaSharpChatCompletion : IChatCompletion, IDisposable
    {
        private readonly Func<LLamaWeights> _modelFunc;
        private readonly LLama.Common.ModelParams _params;
        private LLamaWeights? _model;

        private const string UserRole = "User:";
        private const string AssistantRole = "Assistant:";

        /// <summary>
        /// Create LLamaSharpChatCompletion instance
        /// </summary>
        /// <param name="modelPath"></param>
        public LLamaSharpChatCompletion(LLama.Common.ModelParams @params)
        {
            this._params = @params;
            this._modelFunc = new Func<LLamaWeights>(() =>
            {
                return _model ??= LLamaWeights.LoadFromFile(this._params);
            });
        }

        public LLamaSharpChatCompletion(Func<LLamaWeights> modelFunc, LLama.Common.ModelParams @params)
        {
            this._params = @params;
            this._modelFunc = modelFunc;
        }

        /// <inheritdoc/>
        public ChatHistory CreateNewChat(string? instructions = "")
        {
            var history = new ChatHistory();

            if (instructions != null && !string.IsNullOrEmpty(instructions))
            {
                history.AddSystemMessage(instructions);
            }

            return history;
        }

        /// <inheritdoc/>
        public async Task<IReadOnlyList<IChatResult>> GetChatCompletionsAsync(ChatHistory chat, ChatRequestSettings? requestSettings = null, CancellationToken cancellationToken = default)
        {
            requestSettings ??= new ChatRequestSettings()
            {
                MaxTokens = 256,
                Temperature = 0,
                TopP = 0,
                StopSequences = new List<string> { }
            };

            ChatSession chatSession = CreateChatSession();

            var result = chatSession.ChatAsync(chat.ToLLamaSharpChatHistory(), requestSettings.ToLLamaSharpInferenceParams(), cancellationToken);

            return new List<IChatResult> { new LLamaSharpChatResult(result) }.AsReadOnly();
        }

        /// <inheritdoc/>
        public async IAsyncEnumerable<IChatStreamingResult> GetStreamingChatCompletionsAsync(ChatHistory chat, ChatRequestSettings? requestSettings = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            requestSettings ??= new ChatRequestSettings()
            {
                MaxTokens = 256,
                Temperature = 0,
                TopP = 0,
                StopSequences = new List<string> { }
            };

            var chatSession = CreateChatSession();

            var result = chatSession.ChatAsync(chat.ToLLamaSharpChatHistory(), requestSettings.ToLLamaSharpInferenceParams(), cancellationToken);

            yield return new LLamaSharpChatResult(result);
        }

        private ChatSession CreateChatSession()
        {
            _model ??= _modelFunc.Invoke();
            var context = _model.CreateContext(_params);
            var executor = new InteractiveExecutor(context);
            var chatSession = new ChatSession(executor)
                .WithHistoryTransform(new HistoryTransform())
                .WithOutputTransform(new LLamaTransforms.KeywordTextOutputStreamTransform(new string[] { UserRole, AssistantRole }));
            return chatSession;
        }

        /// <summary>
        /// Dispose LLamaModel
        /// </summary>
        public void Dispose()
        {
            _model?.Dispose();
        }
    }
}
