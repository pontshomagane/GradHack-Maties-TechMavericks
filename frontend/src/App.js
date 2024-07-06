function toggleDetails(detailsId) {
    const detailsElement = document.getElementById(detailsId);
    const displayStyle = detailsElement.style.display;

    if (displayStyle === 'none') {
        detailsElement.style.display = 'block';
    } else {
        detailsElement.style.display = 'none';
    }
}

