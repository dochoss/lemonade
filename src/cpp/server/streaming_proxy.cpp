#include "lemon/streaming_proxy.h"
#include <sstream>
#include <iostream>
#include <chrono>
#include <lemon/utils/aixlog.hpp>

namespace lemon {


namespace {

bool chunk_has_token_payload(const std::string& chunk_text) {
    std::istringstream stream(chunk_text);
    std::string line;
    while (std::getline(stream, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.rfind("data: ", 0) != 0) {
            continue;
        }

        const std::string data = line.substr(6);
        if (data.empty() || data == "[DONE]") {
            continue;
        }

        try {
            const auto parsed = json::parse(data);
            if (!parsed.contains("choices") || !parsed["choices"].is_array()) {
                continue;
            }
            for (const auto& choice : parsed["choices"]) {
                if (choice.contains("delta") && choice["delta"].is_object()) {
                    const auto& delta = choice["delta"];
                    for (const char* key : {"content", "reasoning_content"}) {
                        if (delta.contains(key) && delta[key].is_string() && !delta[key].get<std::string>().empty()) {
                            return true;
                        }
                    }
                }
                if (choice.contains("text") && choice["text"].is_string() && !choice["text"].get<std::string>().empty()) {
                    return true;
                }
            }
        } catch (...) {
            // Ignore non-JSON stream frames.
        }
    }
    return false;
}

} // namespace

void StreamingProxy::forward_sse_stream(
    const std::string& backend_url,
    const std::string& request_body,
    httplib::DataSink& sink,
    std::function<void(const TelemetryData&)> on_complete,
    long timeout_seconds) {

    std::string telemetry_buffer;
    bool stream_error = false;
    bool has_done_marker = false;
    bool saw_first_token = false;
    const auto started = std::chrono::steady_clock::now();
    auto first_token_at = started;

    // Use HttpClient to stream from backend
    auto result = utils::HttpClient::post_stream(
        backend_url,
        request_body,
        [&sink, &telemetry_buffer, &has_done_marker, &saw_first_token, &first_token_at](const char* data, size_t length) {
            // Buffer for telemetry parsing
            telemetry_buffer.append(data, length);

            // Check if this chunk contains [DONE]
            std::string chunk(data, length);
            if (chunk.find("[DONE]") != std::string::npos) {
                has_done_marker = true;
            }
            if (!has_done_marker && !saw_first_token && chunk_has_token_payload(chunk)) {
                saw_first_token = true;
                first_token_at = std::chrono::steady_clock::now();
            }

            // Forward chunk to client immediately
            if (!sink.write(data, length)) {
                return false; // Client disconnected
            }

            return true; // Continue streaming
        },
        {}, // Empty headers map
        timeout_seconds
    );

    if (result.status_code != 200) {
        stream_error = true;
        LOG(ERROR, "StreamingProxy") << "Backend returned error: " << result.status_code << std::endl;
    }

    if (!stream_error) {
        // Ensure [DONE] marker is sent if backend didn't send it
        if (!has_done_marker) {
            LOG(WARNING, "StreamingProxy") << "WARNING: Backend did not send [DONE] marker, adding it" << std::endl;
            const char* done_marker = "data: [DONE]\n\n";
            sink.write(done_marker, strlen(done_marker));
        }

        // Explicitly flush and signal completion
        sink.done();

        LOG(INFO, "Server") << "Streaming completed - 200 OK" << std::endl;

        // Parse telemetry from buffered data
        auto telemetry = parse_telemetry(telemetry_buffer);
        if (telemetry.time_to_first_token <= 0.0 && saw_first_token) {
            telemetry.time_to_first_token = std::chrono::duration<double>(first_token_at - started).count();
        }
        if (telemetry.tokens_per_second <= 0.0 && telemetry.output_tokens > 0) {
            const auto decode_start = saw_first_token ? first_token_at : started;
            const double decode_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - decode_start).count();
            if (decode_seconds > 0.0) {
                telemetry.tokens_per_second = telemetry.output_tokens / decode_seconds;
            }
        }
        telemetry.print();

        if (on_complete) {
            on_complete(telemetry);
        }
    } else {
        // Properly terminate the chunked response even on error
        sink.done();
    }
}

void StreamingProxy::forward_byte_stream(
    const std::string& backend_url,
    const std::string& request_body,
    httplib::DataSink& sink,
    long timeout_seconds) {

    bool stream_error = false;

    // Use HttpClient to stream from backend
    auto result = utils::HttpClient::post_stream(
        backend_url,
        request_body,
        [&sink](const char* data, size_t length) {
            // Forward chunk to client immediately
            if (!sink.write(data, length)) {
                return false; // Client disconnected
            }

            return true; // Continue streaming
        },
        {}, // Empty headers map
        timeout_seconds
    );

    if (result.status_code != 200) {
        stream_error = true;
        LOG(ERROR, "StreamingProxy") << "Backend returned error: " << result.status_code << std::endl;
    }

    if (!stream_error) {
        // Explicitly flush and signal completion
        sink.done();
        LOG(INFO, "Server") << "Streaming completed - 200 OK" << std::endl;
    } else {
        // Properly terminate the chunked response even on error
        sink.done();
    }
}

StreamingProxy::TelemetryData StreamingProxy::parse_telemetry(const std::string& buffer) {
    TelemetryData telemetry;

    std::istringstream stream(buffer);
    std::string line;
    json last_chunk_with_usage;

    while (std::getline(stream, line)) {
        // Handle SSE format (data: ...)
        std::string json_str;
        if (line.find("data: ") == 0) {
            json_str = line.substr(6); // Remove "data: " prefix
        } else if (line.find("ChatCompletionChunk: ") == 0) {
            // FLM debug format
            json_str = line.substr(21); // Remove "ChatCompletionChunk: " prefix
        }

        if (!json_str.empty() && json_str != "[DONE]") {
            try {
                auto chunk = json::parse(json_str);
                // Look for usage or timings in the chunk
                if (chunk.contains("usage") || chunk.contains("timings")) {
                    last_chunk_with_usage = chunk;
                }
            } catch (...) {
                // Skip invalid JSON
            }
        }
    }

    // Extract telemetry from the last chunk with usage data
    if (!last_chunk_with_usage.empty()) {
        try {
            if (last_chunk_with_usage.contains("usage")) {
                auto usage = last_chunk_with_usage["usage"];

                if (usage.contains("prompt_tokens")) {
                    telemetry.input_tokens = usage["prompt_tokens"].get<int>();
                }
                if (usage.contains("completion_tokens")) {
                    telemetry.output_tokens = usage["completion_tokens"].get<int>();
                }

                // FLM format
                if (usage.contains("prefill_duration_ttft")) {
                    telemetry.time_to_first_token = usage["prefill_duration_ttft"].get<double>();
                }
                if (usage.contains("decoding_speed_tps")) {
                    telemetry.tokens_per_second = usage["decoding_speed_tps"].get<double>();
                }
            }

            // Alternative format (timings)
            if (last_chunk_with_usage.contains("timings")) {
                auto timings = last_chunk_with_usage["timings"];

                if (timings.contains("prompt_n")) {
                    telemetry.input_tokens = timings["prompt_n"].get<int>();
                }
                if (timings.contains("predicted_n")) {
                    telemetry.output_tokens = timings["predicted_n"].get<int>();
                }
                if (timings.contains("prompt_ms")) {
                    telemetry.time_to_first_token = timings["prompt_ms"].get<double>() / 1000.0;
                }
                if (timings.contains("predicted_per_second")) {
                    telemetry.tokens_per_second = timings["predicted_per_second"].get<double>();
                }
            }
        } catch (const std::exception& e) {
            LOG(ERROR, "StreamingProxy") << "Error parsing telemetry: " << e.what() << std::endl;
        }
    }

    return telemetry;
}

} // namespace lemon
