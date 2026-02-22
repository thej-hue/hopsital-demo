// components/Register.js
import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

const Register = () => {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const navigate = useNavigate();

  // ------------------ REGISTER ------------------
  const handleRegister = async (e) => {
    e.preventDefault();
    setError("");
    setMessage("");

    try {
      const response = await fetch("http://127.0.0.1:5000/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, email, password }),
      });

      const data = await response.json();

      if (response.ok) {
        setMessage("✅ Registration successful! Redirecting to login...");
        setTimeout(() => {
          navigate("/login"); // redirect to login page
        }, 1500);
      } else {
        setError(data.error || "Registration failed");
      }
    } catch (err) {
      setError("Server error. Make sure backend is running.");
    }
  };

  return (
    <div style={containerStyle}>
      <div style={cardStyle}>
        <h1 style={titleStyle}>Register as Doctor</h1>

        <form onSubmit={handleRegister}>
          <div style={fieldStyle}>
            <label style={labelStyle}>Full Name</label>
            <input
              type="text"
              placeholder="Enter your full name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
              style={inputStyle}
            />
          </div>

          <div style={fieldStyle}>
            <label style={labelStyle}>Email Address</label>
            <input
              type="email"
              placeholder="Enter your email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              style={inputStyle}
            />
          </div>

          <div style={fieldStyle}>
            <label style={labelStyle}>Password</label>
            <input
              type="password"
              placeholder="Enter your password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              style={inputStyle}
            />
          </div>

          {error && <p style={errorStyle}>{error}</p>}
          {message && <p style={messageStyle}>{message}</p>}

          <button type="submit" style={buttonStyle}>
            Register
          </button>

          <p style={toggleTextStyle}>
            Already registered?{" "}
            <span style={linkStyle} onClick={() => navigate("/login")}>
              Login here
            </span>
          </p>
        </form>
      </div>
    </div>
  );
};

export default Register;

// ---------------- STYLES ----------------
const containerStyle = {
  fontFamily: "Poppins, sans-serif",
  background: "linear-gradient(135deg, #1E88E5 0%, #42A5F5 100%)",
  height: "100vh",
  display: "flex",
  justifyContent: "center",
  alignItems: "center",
};

const cardStyle = {
  backgroundColor: "#fff",
  padding: "50px 40px",
  borderRadius: "20px",
  boxShadow: "0 10px 30px rgba(0,0,0,0.15)",
  width: "380px",
  textAlign: "center",
};

const titleStyle = {
  color: "#1565C0",
  marginBottom: "25px",
  fontSize: "24px",
  fontWeight: "700",
};

const fieldStyle = {
  marginBottom: "20px",
  textAlign: "left",
};

const inputStyle = {
  width: "100%",
  padding: "12px",
  border: "1px solid #ddd",
  borderRadius: "8px",
  fontSize: "15px",
  outline: "none",
};

const labelStyle = {
  color: "#555",
  fontWeight: "500",
  fontSize: "14px",
  display: "block",
  marginBottom: "6px",
};

const buttonStyle = {
  width: "100%",
  padding: "12px",
  backgroundColor: "#1E88E5",
  color: "#fff",
  border: "none",
  borderRadius: "8px",
  fontSize: "15px",
  fontWeight: "600",
  cursor: "pointer",
};

const errorStyle = {
  color: "#D32F2F",
  marginBottom: "15px",
  fontSize: "14px",
  fontWeight: "500",
};

const messageStyle = {
  color: "green",
  marginBottom: "15px",
  fontSize: "14px",
  fontWeight: "500",
};

const toggleTextStyle = {
  marginTop: "15px",
  fontSize: "14px",
  color: "#555",
};

const linkStyle = {
  color: "#1E88E5",
  cursor: "pointer",
  fontWeight: "bold",
};