/*
 * TLSTMClient.cc
 *
 *  Created on: Aug 3, 2025
 *      Author: imasha
 */
#include <omnetpp.h>
#include <iostream>
#include <vector>
#include <arpa/inet.h>
#include <unistd.h>

using namespace omnetpp;

class TLSTMClient : public cSimpleModule
{
  protected:
    const simtime_t interval = SimTime(0.05, SIMTIME_S);
    virtual void initialize() override;
    virtual void handleMessage(cMessage *msg) override;
    std::vector<float> prepareInput();
    std::vector<float> requestPrediction(const std::vector<float>& input);
};

Define_Module(TLSTMClient);

void TLSTMClient::initialize() {
    EV << "Initializing TLSTMClient...\n";
    cMessage* predictMsg = new cMessage("predict");
    predictMsg->setKind(1);  // optional for debugging
    scheduleAt(simTime() + interval, predictMsg);
    //scheduleAt(simTime() + interval, new cMessage("predict"));
}

void TLSTMClient::handleMessage(cMessage *msg) {
    EV << "[TLSTMClient] handleMessage() called with message: " << msg->getName() << "\n";

        std::vector<float> input = prepareInput();
        std::vector<float> prediction = requestPrediction(input);

        EV << "?? Predicted Packet Rate: " << prediction[0]
           << " | Byte Rate: " << prediction[1] << "\n";

        scheduleAt(simTime() + interval, new cMessage("predict"));
        delete msg;
}

std::vector<float> TLSTMClient::prepareInput()
{
    const int timesteps = 10;
    const int input_dim = 50;

    std::vector<float> input(timesteps * input_dim);
    for (int t = 0; t < timesteps; t++) {
            input[t * input_dim + 0] = lastPacketRate[t];
            input[t * input_dim + 1] = lastByteRate[t];
            // ...
            input[t * input_dim + timeDeltaIndex] = timeDelta[t];
        }
        return input;
}

std::vector<float> TLSTMClient::requestPrediction(const std::vector<float>& input)
{
    std::vector<float> prediction(2, 0.0f);
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        EV << "Socket creation failed\n";
        return prediction;
    }

    sockaddr_in serverAddr{};
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(5005);
    inet_pton(AF_INET, "172.24.242.178", &serverAddr.sin_addr);

    struct timeval timeout;
    timeout.tv_sec = 3;
    timeout.tv_usec = 0;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

    if (connect(sock, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
        EV << "? TLSTMClient: Could not connect to prediction server at 127.0.0.1:5005\n";
        EV << "Errno: " << strerror(errno) << "\n";
        close(sock);
        return prediction;
    } else {
        EV << "? TLSTMClient: Connected to prediction server\n";
    }

    ::send(sock, input.data(), input.size() * sizeof(float), 0);

    float buffer[2];
    char* dataPtr = reinterpret_cast<char*>(buffer);
    int totalBytes = sizeof(buffer);
    int received = 0;

    while (received < totalBytes) {
        int r = recv(sock, dataPtr + received, totalBytes - received, 0);
        if (r <= 0) {
            EV << "? Failed to receive prediction data\n";
            close(sock);
            return prediction;
        }
        received += r;
    }

    prediction[0] = buffer[0];
    prediction[1] = buffer[1];

    close(sock);
    return prediction;
}




