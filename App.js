import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');

  const handleInputChange = (event) => {
    setInputText(event.target.value);
  };

  const handleSend = async () => {
    const userInputText = inputText.trim();
    if (!userInputText) return;

    const newMessage = { id: Date.now(), text: userInputText, sender: 'user' };
    setMessages(messages.concat(newMessage));
    setInputText('');

    try {
      const response = await axios.post('http://localhost:5000/generate_insights', {
        texts: [userInputText],
      });

      const botResponse = response.data.insights.map((insight, index) => ({
        id: newMessage.id + index + 1,
        text: `Summary: ${insight.summary}\nGoals: ${insight.goals.join(', ')}`,
        sender: 'bot',
        image: insight.image_base64 ? `data:image/png;base64,${insight.image_base64}` : null,
      }));

      setMessages(messages.concat(newMessage).concat(botResponse));
    } catch (error) {
      console.error('Error fetching insights:', error);
      const errorMessage = { id: Date.now(), text: 'Sorry, could not fetch insights.', sender: 'bot' };
      setMessages(messages.concat(newMessage).concat(errorMessage));
    }
  };

  return (
    <div className="App">
      <div className="chat-window">
        {messages.map((msg) => (
          <div key={msg.id} className={`message ${msg.sender}`}>
            <p>{msg.text}</p>
            {msg.image && <img src={msg.image} alt="Insight" />}
          </div>
        ))}
      </div>
      <div className="input-area">
        <input type="text" value={inputText} onChange={handleInputChange} />
        <button onClick={handleSend}>Send</button>
      </div>
    </div>
  );
}

export default App;
