import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import Particle from "../Particle";
import homeLogo from "../../Assets/danny_picture.png";
import Tilt from "react-parallax-tilt";

function About() {
  return (
    <Container fluid className="about-section">
      {/* Particle background effect  */}
      <Particle />
      <Container>
      <Row>
            <Col md={7} className="home-header">
            <h1 style={{ paddingBottom: 15 }} className="heading">
              G'Day, Mate!{" "}
              <span className="kangaroo">🦘</span>
            </h1>

              <h1 className="heading-name">
                I'm
                <strong className="main-name"> Danny Nguyen</strong>
              </h1>
            </Col>

            <Col md={5} style={{ paddingBottom: 20 }}>
              <Tilt>
                <img
                  src={homeLogo}
                  alt="home pic"
                  className="img-fluid"
                  style={{ maxHeight: "450px" }}
                />
              </Tilt>
            </Col>
          </Row>

          <Row>
            <Col md={7} className="home-header">
            <h1 style={{ paddingBottom: 15 }} className="heading">
              G'Day, Mate!{" "}
              <span className="kangaroo">🦘</span>
            </h1>

              <h1 className="heading-name">
                I'm
                <strong className="main-name"> Danny Nguyen</strong>
              </h1>
            </Col>

            <Col md={5} style={{ paddingBottom: 20 }}>
              <Tilt>
                <img
                  src={homeLogo}
                  alt="home pic"
                  className="img-fluid"
                  style={{ maxHeight: "450px" }}
                />
              </Tilt>
            </Col>
          </Row>



          <Row>
            <Col md={7} className="home-header">
            <h1 style={{ paddingBottom: 15 }} className="heading">
              G'Day, Mate!{" "}
              <span className="kangaroo">🦘</span>
            </h1>

              <h1 className="heading-name">
                I'm
                <strong className="main-name"> Danny Nguyen</strong>
              </h1>
            </Col>

            <Col md={5} style={{ paddingBottom: 20 }}>
              <Tilt>
                <img
                  src={homeLogo}
                  alt="home pic"
                  className="img-fluid"
                  style={{ maxHeight: "450px" }}
                />
              </Tilt>
            </Col>
          </Row>




      </Container>
    </Container>
  );
}

export default About;
